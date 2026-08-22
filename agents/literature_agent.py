"""
Literature Survey Agent — backend logic for IsoScreenAI Page 3.

Pipeline:
  1. search_pubmed()       -> PMIDs for a research question
  2. fetch_articles()      -> title/year/DOI/abstract per PMID
  3. rank_by_relevance()   -> one Groq call scores + sorts papers by relevance
  4. generate_report()     -> one Groq call writes the full Markdown report
  5. run_literature_survey() -> orchestrates 1-4 for the Streamlit page
"""

from __future__ import annotations

import json
import logging
import re
import time

from Bio import Entrez
from groq import Groq

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Entrez.email = getattr(config, "ENTREZ_EMAIL", "isoscreenai@example.com")
if getattr(config, "NCBI_API_KEY", None):
    Entrez.api_key = config.NCBI_API_KEY

_GROQ_MODEL = getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")
_GROQ_API_KEY = getattr(config, "GROQ_API_KEY", None)
_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else Groq()


class LiteratureAgentError(Exception):
    """Raised for any recoverable failure in the literature survey pipeline."""


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

def _groq_call(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    try:
        resp = _client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e).lower()
        if "rate_limit" in msg or "429" in msg:
            raise LiteratureAgentError("Groq rate limit hit. Try again shortly or reduce article count.")
        raise LiteratureAgentError(f"Groq call failed: {e}")


def _groq_json(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> list | dict:
    raw = _groq_call(system_prompt, user_prompt, max_tokens)
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise LiteratureAgentError("The model returned malformed output. Try again.")


# ---------------------------------------------------------------------------
# PubMed fetch
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 15) -> list[str]:
    for attempt in range(3):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            logger.warning("esearch attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.5)
    raise LiteratureAgentError("PubMed search failed after retries.")


def _doi_and_url(article_xml: dict, pubmed_data: dict, pmid: str) -> tuple[str | None, str]:
    doi = None
    for eloc in article_xml.get("ELocationID", []):
        if getattr(eloc, "attributes", {}).get("EIdType") == "doi":
            doi = str(eloc)
    if not doi:
        for aid in pubmed_data.get("ArticleIdList", []):
            if getattr(aid, "attributes", {}).get("IdType") == "doi":
                doi = str(aid)
    url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return doi, url


def fetch_articles(pmids: list[str]) -> list[dict]:
    """Return a list of {pmid, title, abstract, year, doi, url} dicts."""
    if not pmids:
        return []
    try:
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        raise LiteratureAgentError(f"PubMed fetch failed: {e}")

    articles = []
    for record in records.get("PubmedArticle", []):
        try:
            medline = record["MedlineCitation"]
            art = medline["Article"]
            pmid = str(medline["PMID"])
            title = str(art.get("ArticleTitle", "")).strip() or "(no title)"
            abstract = " ".join(
                str(p) for p in art.get("Abstract", {}).get("AbstractText", [])
            ).strip() or "(no abstract available)"
            year = None
            try:
                year = art["Journal"]["JournalIssue"]["PubDate"].get("Year")
            except Exception:
                pass
            doi, url = _doi_and_url(art, record.get("PubmedData", {}), pmid)
            articles.append(
                {"pmid": pmid, "title": title, "abstract": abstract, "year": year, "doi": doi, "url": url}
            )
        except Exception as e:
            logger.warning("Skipping malformed PubMed record: %s", e)
    return articles


# ---------------------------------------------------------------------------
# Relevance ranking (single Groq call)
# ---------------------------------------------------------------------------

_RANK_PROMPT = """You are a research librarian. Given a research question and a list of \
papers (PMID, title, abstract), score each paper's relevance to the question from 0-10 \
(10 = directly answers the question, 0 = unrelated) with a one-sentence reason.

Return ONLY a JSON array, one object per paper:
[{"pmid": "...", "relevance_score": 0-10, "reason": "..."}]
"""


def rank_by_relevance(query: str, articles: list[dict]) -> list[dict]:
    """Score + sort articles highest-relevance-first via one batched Groq call."""
    if not articles:
        return []

    blocks = [f"PMID: {a['pmid']}\nTitle: {a['title']}\nAbstract: {a['abstract'][:1200]}" for a in articles]
    user_prompt = f"Research question: {query}\n\n" + "\n\n---\n\n".join(blocks)

    try:
        scores = _groq_json(_RANK_PROMPT, user_prompt, max_tokens=1500)
    except LiteratureAgentError as e:
        logger.warning("Relevance ranking failed, keeping PubMed order: %s", e)
        scores = []

    score_map = {str(s.get("pmid", "")): s for s in scores if isinstance(s, dict)}
    for a in articles:
        s = score_map.get(a["pmid"])
        a["relevance_score"] = float(s.get("relevance_score", 0)) if s else 0.0
        a["relevance_reason"] = s.get("reason", "Not scored.") if s else "Not scored."

    articles.sort(key=lambda a: a["relevance_score"], reverse=True)
    return articles


# ---------------------------------------------------------------------------
# Report generation (single Groq call)
# ---------------------------------------------------------------------------

_REPORT_PROMPT = """You are a senior research scientist writing a literature review for \
a drug-discovery research question. You are given the research question and a set of \
PubMed papers (title, year, PMID, relevance score, link, abstract), ordered most to \
least relevant. Base everything only on the abstracts provided.

Write a complete Markdown report with exactly these sections:

# Literature Review: <research question>

## Overview
2-4 sentence summary of the current state of evidence.

## Included Studies
A Markdown table: | Title | Year | Relevance | Link | (use each paper's given link).

## Key Findings by Study
For each paper, most relevant first, a short subsection covering: study type, methods, \
key findings, and limitations.

## Gaps & Unanswered Questions
Bullet list of what the literature does not yet address.

## Conflicting Findings or Theories
Bullet list of contradictions across papers, citing PMIDs. State "None identified" if none.

## References
Numbered list: PMID, title, year, link.

Output ONLY the Markdown report — no commentary before or after it.
"""


def generate_report(query: str, ranked_articles: list[dict]) -> str:
    if not ranked_articles:
        return f"# Literature Review: {query}\n\nNo PubMed articles were found for this query."

    blocks = [
        f"PMID: {a['pmid']} | {a['title']} ({a.get('year', 'n.d.')})\n"
        f"Relevance: {a['relevance_score']}/10 | Link: {a['url']}\n"
        f"Abstract: {a['abstract'][:2000]}"
        for a in ranked_articles
    ]
    user_prompt = f"Research question: {query}\n\n" + "\n\n---\n\n".join(blocks)
    return _groq_call(_REPORT_PROMPT, user_prompt, max_tokens=4000)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_literature_survey(query: str, max_results: int = 15) -> dict:
    """Search -> fetch -> rank -> report. Returns {query, articles, report_md}."""
    query = (query or "").strip()
    if not query:
        raise LiteratureAgentError("Please enter a research question.")

    pmids = search_pubmed(query, max_results=max_results)
    if not pmids:
        raise LiteratureAgentError(f"No PubMed results found for '{query}'.")

    articles = fetch_articles(pmids)
    ranked = rank_by_relevance(query, articles)
    report_md = generate_report(query, ranked)

    return {"query": query, "articles": ranked, "report_md": report_md}
