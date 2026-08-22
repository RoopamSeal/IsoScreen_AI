"""
Literature Survey Agent
------------------------
Backend logic for IsoScreenAI Page 3: a staged, semi-systematic literature
review pipeline built around a researcher's free-text research question.

Architecture
------------
Each of the six requirements maps to one explicit, independently callable
stage. Stages are pure functions operating on dataclasses, so each can be
tested, cached, or re-run in isolation instead of hiding everything behind
one opaque "run everything" call:

    1. search_pubmed()            -> PMIDs for the research question
    2. fetch_articles()           -> Article records (title, year, DOI, abstract)
    3. dedupe_articles()          -> duplicate-free Article list
       screen_abstracts()         -> LLM relevance scoring per article
       select_included()          -> deterministic, re-runnable local filter
                                      (no LLM call) so a UI can move the
                                      relevance-threshold slider freely
                                      without re-billing the API
    4. extract_paper_details()    -> methods / findings / datasets / limitations
                                      per included paper
    5. synthesize_review()        -> cross-paper gaps, conflicts, open questions
    6. build_markdown_report()    -> single downloadable Markdown deliverable

run_full_pipeline() is a thin convenience wrapper that chains all of the
above for non-interactive / scripted use; the Streamlit page calls the
staged functions directly so it can show intermediate results and avoid
re-running expensive LLM stages when only a filter changes.

Cost/latency notes:
- Screening batches abstracts (default 5/call) into one JSON-array call
  rather than one call per paper.
- Extraction only runs on the *included* subset (post-screening), and is
  capped by `max_papers_to_extract` to bound LLM spend on broad queries.
- Synthesis operates on the compact extracted fields, not raw abstracts,
  to keep the final synthesis prompt small.
- All extraction is abstract-level (PubMed's API does not reliably expose
  full text), so "methods"/"datasets"/"limitations" reflect what authors
  reported in their abstract, not the full paper.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable, Optional

from Bio import Entrez
from groq import Groq

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

Entrez.email = getattr(config, "ENTREZ_EMAIL", "isoscreenai@example.com")
_NCBI_API_KEY = getattr(config, "NCBI_API_KEY", None)
if _NCBI_API_KEY:
    Entrez.api_key = _NCBI_API_KEY

_GROQ_API_KEY = getattr(config, "GROQ_API_KEY", None)
_GROQ_MODEL = getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")
_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else Groq()

ProgressCallback = Optional[Callable[[str, float], None]]


def _report_progress(cb: ProgressCallback, message: str, fraction: float) -> None:
    if cb:
        try:
            cb(message, fraction)
        except Exception:  # never let a UI callback break the pipeline
            pass


class LiteratureAgentError(Exception):
    """Raised for any recoverable failure in the literature review pipeline."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: Optional[str]
    authors: str
    doi: Optional[str]
    url: str  # DOI link if available, else PubMed link


@dataclass
class ScreenedArticle(Article):
    relevance_score: float = 0.0       # 0-10, LLM-assigned
    screening_reason: str = ""
    recommended_include: bool = False  # LLM's own include/exclude call


@dataclass
class ExtractedPaper:
    pmid: str
    title: str
    year: Optional[str]
    study_type: str
    methods: str
    key_findings: str
    datasets: str
    sample_size: str
    limitations: str


@dataclass
class Conflict:
    topic: str
    description: str
    pmids: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    overall_synthesis: str
    gaps: list[str] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


def to_records(items: list) -> list[dict]:
    """Convert a list of dataclass instances to plain dicts (for pd.DataFrame)."""
    return [asdict(i) for i in items]


# ---------------------------------------------------------------------------
# Shared Groq JSON-call helper (used by screening / extraction / synthesis)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _salvage_json(text: str, container: str = "{") -> Optional[dict | list]:
    """Best-effort recovery if the model wraps JSON in extra prose."""
    opener, closer = ("{", "}") if container == "{" else ("[", "]")
    match = re.search(rf"\{re.escape(opener)}.*\{re.escape(closer)}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _groq_json_call(
    system_prompt: str,
    user_prompt: str,
    *,
    expect_array: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> dict | list:
    """Call Groq expecting a raw-JSON response; parse defensively."""
    try:
        response = _client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate_limit" in msg or "429" in msg:
            raise LiteratureAgentError(
                "Groq rate limit / token limit hit. Try again shortly, or reduce "
                "the number of articles / batch size."
            )
        raise LiteratureAgentError(f"Groq call failed: {e}")

    raw = response.choices[0].message.content
    cleaned = _strip_code_fences(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        salvaged = _salvage_json(cleaned, container="[" if expect_array else "{")
        if salvaged is None:
            logger.warning("Could not parse Groq JSON output; raw: %s", raw[:500])
            raise LiteratureAgentError(
                "The model returned malformed output for this step. Try again."
            )
        return salvaged


# ---------------------------------------------------------------------------
# Stage 1-2: Search + Fetch
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 15, retries: int = 2) -> list[str]:
    """Return PMIDs matching the research question."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            last_err = e
            logger.warning("Entrez esearch failed (attempt %d): %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise LiteratureAgentError(f"PubMed search failed after retries: {last_err}")


def _extract_doi(article_xml: dict, pubmed_data: dict) -> Optional[str]:
    """Look for a DOI in ELocationID first, then PubmedData/ArticleIdList."""
    try:
        for eloc in article_xml.get("ELocationID", []):
            if getattr(eloc, "attributes", {}).get("EIdType") == "doi":
                return str(eloc)
    except Exception:
        pass
    try:
        for aid in pubmed_data.get("ArticleIdList", []):
            if getattr(aid, "attributes", {}).get("IdType") == "doi":
                return str(aid)
    except Exception:
        pass
    return None


def _format_authors(author_list: list) -> str:
    names = []
    for a in author_list[:3]:
        last = a.get("LastName", "")
        init = a.get("Initials", "")
        if last:
            names.append(f"{last} {init}".strip())
    if not names:
        return "Unknown authors"
    suffix = ", et al." if len(author_list) > 3 else ""
    return ", ".join(names) + suffix


def fetch_articles(pmids: list[str]) -> list[Article]:
    """Fetch title/abstract/year/journal/authors/DOI for a list of PMIDs."""
    if not pmids:
        return []
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        raise LiteratureAgentError(f"PubMed fetch failed: {e}")

    articles: list[Article] = []
    for record in records.get("PubmedArticle", []):
        try:
            medline = record["MedlineCitation"]
            art = medline["Article"]
            pubmed_data = record.get("PubmedData", {})
            pmid = str(medline["PMID"])
            title = str(art.get("ArticleTitle", "")).strip() or "(no title available)"

            abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts).strip() or "(no abstract available)"

            year = None
            try:
                year = art["Journal"]["JournalIssue"]["PubDate"].get("Year")
            except Exception:
                pass

            journal = str(art.get("Journal", {}).get("Title", "")).strip()
            authors = _format_authors(art.get("AuthorList", []))
            doi = _extract_doi(art, pubmed_data)
            url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            articles.append(
                Article(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    journal=journal,
                    year=year,
                    authors=authors,
                    doi=doi,
                    url=url,
                )
            )
        except Exception as e:
            logger.warning("Skipping malformed PubMed record: %s", e)
            continue

    return articles


# ---------------------------------------------------------------------------
# Stage 3a: Deduplication
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def dedupe_articles(articles: list[Article], title_similarity_threshold: float = 0.92) -> tuple[list[Article], int]:
    """Remove duplicates by exact DOI match, then fuzzy title match.

    Returns (deduped_articles, number_removed).
    """
    seen_dois: set[str] = set()
    kept: list[Article] = []
    kept_norm_titles: list[str] = []
    removed = 0

    for a in articles:
        if a.doi:
            doi_key = a.doi.lower().strip()
            if doi_key in seen_dois:
                removed += 1
                continue
        norm_title = _normalize_title(a.title)
        is_dupe = False
        if a.doi and a.doi.lower().strip() in seen_dois:
            is_dupe = True
        else:
            for kept_title in kept_norm_titles:
                if difflib.SequenceMatcher(None, norm_title, kept_title).ratio() >= title_similarity_threshold:
                    is_dupe = True
                    break
        if is_dupe:
            removed += 1
            continue

        kept.append(a)
        kept_norm_titles.append(norm_title)
        if a.doi:
            seen_dois.add(a.doi.lower().strip())

    return kept, removed


# ---------------------------------------------------------------------------
# Stage 3b: Abstract screening (LLM relevance scoring)
# ---------------------------------------------------------------------------

_SCREENING_SYSTEM_PROMPT = """You are a systematic-review screening assistant. \
Given a research question and a batch of PubMed abstracts, score each abstract's \
relevance to the research question.

Return ONLY a valid JSON array (no markdown fences, no commentary), one object per \
abstract, in this exact schema:

[
  {
    "pmid": "string, must exactly match the PMID given",
    "relevance_score": 0-10 (integer, 10 = directly and centrally addresses the question),
    "include": true/false (true if relevance_score >= 6),
    "reason": "one sentence explaining the score"
  }
]

Score strictly: a paper that merely mentions a related keyword without addressing the \
question should score low (0-3). A paper that directly investigates the question's \
subject should score high (7-10).
"""


def screen_abstracts(
    research_question: str,
    articles: list[Article],
    batch_size: int = 5,
    progress_callback: ProgressCallback = None,
) -> list[ScreenedArticle]:
    """Score every article's relevance to the research question via batched LLM calls."""
    if not articles:
        return []

    screened: list[ScreenedArticle] = []
    batches = [articles[i : i + batch_size] for i in range(0, len(articles), batch_size)]

    for b_idx, batch in enumerate(batches):
        _report_progress(
            progress_callback,
            f"Screening abstracts (batch {b_idx + 1}/{len(batches)})...",
            b_idx / max(len(batches), 1),
        )
        blocks = [f'PMID: {a.pmid}\nTitle: {a.title}\nAbstract: {a.abstract[:1500]}' for a in batch]
        user_prompt = (
            f"Research question: {research_question}\n\n"
            f"Score the following {len(batch)} abstracts:\n\n" + "\n\n---\n\n".join(blocks)
        )

        try:
            results = _groq_json_call(
                _SCREENING_SYSTEM_PROMPT, user_prompt, expect_array=True, max_tokens=1200
            )
        except LiteratureAgentError as e:
            logger.warning("Screening batch %d failed: %s", b_idx, e)
            results = []

        scores_by_pmid = {}
        if isinstance(results, list):
            for r in results:
                pmid = str(r.get("pmid", "")).strip()
                if pmid:
                    scores_by_pmid[pmid] = r

        for a in batch:
            r = scores_by_pmid.get(a.pmid)
            if r:
                score = float(r.get("relevance_score", 0) or 0)
                screened.append(
                    ScreenedArticle(
                        **asdict(a),
                        relevance_score=score,
                        screening_reason=r.get("reason", ""),
                        recommended_include=bool(r.get("include", score >= 6)),
                    )
                )
            else:
                # Screening failed for this article; flag it for manual review
                # rather than silently dropping it.
                screened.append(
                    ScreenedArticle(
                        **asdict(a),
                        relevance_score=0.0,
                        screening_reason="Automatic screening failed for this article; review manually.",
                        recommended_include=False,
                    )
                )

    _report_progress(progress_callback, "Screening complete.", 1.0)
    return screened


def select_included(
    screened: list[ScreenedArticle],
    relevance_threshold: float = 6.0,
    max_papers: int = 12,
) -> list[ScreenedArticle]:
    """Deterministic, local (no-LLM-call) filter — safe to re-run on every UI interaction.

    Selects articles at/above the threshold, capped to the top `max_papers` by score
    so a broad query can't blow up downstream extraction cost.
    """
    eligible = [s for s in screened if s.relevance_score >= relevance_threshold]
    eligible.sort(key=lambda s: s.relevance_score, reverse=True)
    return eligible[:max_papers]


# ---------------------------------------------------------------------------
# Stage 4: Per-paper structured extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a biomedical literature analyst. Extract \
structured information from a single paper's abstract for a systematic literature \
review. Base your answer only on what is stated or clearly implied in the abstract \
provided — do not invent details the abstract does not support.

Return ONLY valid JSON (no markdown fences, no commentary) in this exact schema:

{
  "study_type": "short label, e.g. 'RCT', 'cohort study', 'in vitro', 'computational/in silico', 'review', 'case report'",
  "methods": "1-2 sentence summary of the methodology/approach used",
  "key_findings": "1-2 sentence summary of the main results",
  "datasets": "datasets, cohorts, cell lines, or data sources used; write 'Not specified' if the abstract does not say",
  "sample_size": "sample size / n, if stated; write 'Not specified' if absent",
  "limitations": "limitations stated or clearly implied by the abstract; write 'Not stated in abstract' if none are given"
}
"""


def extract_paper_details(
    articles: list[ScreenedArticle] | list[Article],
    progress_callback: ProgressCallback = None,
) -> list[ExtractedPaper]:
    """Run one extraction call per paper (only call this on the included subset)."""
    extracted: list[ExtractedPaper] = []
    total = len(articles) or 1

    for i, a in enumerate(articles):
        _report_progress(
            progress_callback, f"Extracting details ({i + 1}/{len(articles)}): {a.title[:60]}...", i / total
        )
        user_prompt = f"Title: {a.title}\n\nAbstract: {a.abstract[:2500]}"
        try:
            data = _groq_json_call(_EXTRACTION_SYSTEM_PROMPT, user_prompt, max_tokens=600)
        except LiteratureAgentError as e:
            logger.warning("Extraction failed for PMID %s: %s", a.pmid, e)
            data = {}

        extracted.append(
            ExtractedPaper(
                pmid=a.pmid,
                title=a.title,
                year=a.year,
                study_type=data.get("study_type", "Not determined"),
                methods=data.get("methods", "Extraction failed for this paper."),
                key_findings=data.get("key_findings", "Extraction failed for this paper."),
                datasets=data.get("datasets", "Not specified"),
                sample_size=data.get("sample_size", "Not specified"),
                limitations=data.get("limitations", "Not stated in abstract"),
            )
        )

    _report_progress(progress_callback, "Extraction complete.", 1.0)
    return extracted


# ---------------------------------------------------------------------------
# Stage 5: Cross-paper synthesis (gaps, conflicts, open questions)
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """You are a senior research scientist writing the \
discussion section of a literature review. You will be given structured summaries \
(methods, findings, limitations) extracted from multiple papers addressing the same \
research question.

Return ONLY valid JSON (no markdown fences, no commentary) in this exact schema:

{
  "overall_synthesis": "2-4 sentence narrative synthesizing the current state of evidence",
  "gaps": ["short bullet describing a missing type of data / unstudied population / methodological gap", "..."],
  "conflicts": [
    {
      "topic": "short label for the point of disagreement",
      "description": "1-2 sentences describing the conflicting findings or theories",
      "pmids": ["pmid1", "pmid2"]
    }
  ],
  "open_questions": ["short bullet phrased as an open research question", "..."]
}

Only report conflicts that are genuinely supported by contradicting findings/claims \
across the provided papers — do not fabricate disagreement where papers are simply \
studying different things. If there are no clear conflicts, return an empty list.
"""


def synthesize_review(research_question: str, extracted: list[ExtractedPaper]) -> SynthesisResult:
    if not extracted:
        return SynthesisResult(
            overall_synthesis="No papers were extracted, so no synthesis could be generated.",
            gaps=[],
            conflicts=[],
            open_questions=[],
        )

    blocks = []
    for p in extracted:
        blocks.append(
            f"PMID: {p.pmid} | {p.title} ({p.year})\n"
            f"Study type: {p.study_type}\n"
            f"Methods: {p.methods}\n"
            f"Findings: {p.key_findings}\n"
            f"Limitations: {p.limitations}"
        )
    user_prompt = (
        f"Research question: {research_question}\n\n"
        f"Extracted summaries from {len(extracted)} included papers:\n\n"
        + "\n\n---\n\n".join(blocks)
    )

    try:
        data = _groq_json_call(_SYNTHESIS_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    except LiteratureAgentError as e:
        logger.warning("Synthesis failed: %s", e)
        return SynthesisResult(
            overall_synthesis=f"Synthesis could not be generated: {e}",
            gaps=[],
            conflicts=[],
            open_questions=[],
        )

    conflicts = [
        Conflict(
            topic=c.get("topic", "Untitled"),
            description=c.get("description", ""),
            pmids=c.get("pmids", []),
        )
        for c in data.get("conflicts", [])
    ]

    return SynthesisResult(
        overall_synthesis=data.get("overall_synthesis", ""),
        gaps=data.get("gaps", []),
        conflicts=conflicts,
        open_questions=data.get("open_questions", []),
    )


# ---------------------------------------------------------------------------
# Stage 6: Downloadable report
# ---------------------------------------------------------------------------

def build_markdown_report(
    research_question: str,
    all_articles: list[Article],
    duplicates_removed: int,
    screened: list[ScreenedArticle],
    included: list[ScreenedArticle],
    extracted: list[ExtractedPaper],
    synthesis: SynthesisResult,
) -> str:
    lines: list[str] = []
    lines.append(f"# Literature Review: {research_question}")
    lines.append(f"*Generated by IsoScreenAI on {date.today().isoformat()}*\n")

    lines.append("## 1. Search & Screening Summary")
    lines.append(f"- Articles fetched from PubMed: **{len(all_articles)}**")
    lines.append(f"- Duplicates removed: **{duplicates_removed}**")
    lines.append(f"- Articles screened for relevance: **{len(screened)}**")
    lines.append(f"- Articles selected as high-relevance (included): **{len(included)}**\n")

    lines.append("## 2. Included Studies")
    lines.append("| Title | Year | Relevance | Link |")
    lines.append("|---|---|---|---|")
    for s in included:
        title_escaped = s.title.replace("|", "-")
        lines.append(f"| {title_escaped} | {s.year or 'n.d.'} | {s.relevance_score:.0f}/10 | [Link]({s.url}) |")
    lines.append("")

    lines.append("## 3. Extracted Study Details")
    for p in extracted:
        lines.append(f"### {p.title} ({p.year or 'n.d.'})")
        lines.append(f"- **PMID:** {p.pmid}")
        lines.append(f"- **Study type:** {p.study_type}")
        lines.append(f"- **Methods:** {p.methods}")
        lines.append(f"- **Key findings:** {p.key_findings}")
        lines.append(f"- **Datasets:** {p.datasets}")
        lines.append(f"- **Sample size:** {p.sample_size}")
        lines.append(f"- **Limitations:** {p.limitations}\n")

    lines.append("## 4. Synthesis")
    lines.append(synthesis.overall_synthesis + "\n")

    lines.append("### Gaps in the Current Literature")
    if synthesis.gaps:
        for g in synthesis.gaps:
            lines.append(f"- {g}")
    else:
        lines.append("- No clear gaps identified from the included papers.")
    lines.append("")

    lines.append("### Conflicting Findings / Theories")
    if synthesis.conflicts:
        for c in synthesis.conflicts:
            pmid_str = ", ".join(c.pmids) if c.pmids else "n/a"
            lines.append(f"- **{c.topic}:** {c.description} (PMIDs: {pmid_str})")
    else:
        lines.append("- No clear conflicts identified across the included papers.")
    lines.append("")

    lines.append("### Open Research Questions")
    if synthesis.open_questions:
        for q in synthesis.open_questions:
            lines.append(f"- {q}")
    else:
        lines.append("- None identified.")
    lines.append("")

    lines.append("## 5. Excluded Studies (Screened Out)")
    excluded = [s for s in screened if s not in included]
    if excluded:
        lines.append("| Title | Year | Relevance | Reason |")
        lines.append("|---|---|---|---|")
        for s in excluded:
            title_escaped = s.title.replace("|", "-")
            lines.append(f"| {title_escaped} | {s.year or 'n.d.'} | {s.relevance_score:.0f}/10 | {s.screening_reason} |")
    else:
        lines.append("_All fetched, deduplicated articles were included._")
    lines.append("")

    lines.append("## References")
    for i, p in enumerate(extracted, start=1):
        match = next((s for s in included if s.pmid == p.pmid), None)
        url = match.url if match else f"https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/"
        lines.append(f"{i}. PMID {p.pmid}. {p.title} ({p.year or 'n.d.'}). {url}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience orchestrator (for scripted / non-UI use)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    research_question: str,
    max_results: int = 15,
    relevance_threshold: float = 6.0,
    max_papers_to_extract: int = 12,
    progress_callback: ProgressCallback = None,
) -> dict:
    """Chains all six stages. The Streamlit page calls the stages individually
    instead of this, so it can render intermediate results and avoid re-running
    expensive stages on every widget interaction — but this is useful for
    scripts, notebooks, or tests.
    """
    research_question = (research_question or "").strip()
    if not research_question:
        raise LiteratureAgentError("Please enter a research question.")

    _report_progress(progress_callback, "Searching PubMed...", 0.05)
    pmids = search_pubmed(research_question, max_results=max_results)
    if not pmids:
        raise LiteratureAgentError(f"No PubMed results found for '{research_question}'.")

    _report_progress(progress_callback, "Fetching articles...", 0.15)
    fetched = fetch_articles(pmids)

    deduped, removed = dedupe_articles(fetched)

    screened = screen_abstracts(research_question, deduped, progress_callback=progress_callback)
    included = select_included(screened, relevance_threshold, max_papers_to_extract)

    extracted = extract_paper_details(included, progress_callback=progress_callback)

    _report_progress(progress_callback, "Synthesizing findings...", 0.9)
    synthesis = synthesize_review(research_question, extracted)

    report_md = build_markdown_report(
        research_question, fetched, removed, screened, included, extracted, synthesis
    )
    _report_progress(progress_callback, "Done.", 1.0)

    return {
        "research_question": research_question,
        "fetched": fetched,
        "duplicates_removed": removed,
        "screened": screened,
        "included": included,
        "extracted": extracted,
        "synthesis": synthesis,
        "report_md": report_md,
    }
