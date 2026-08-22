"""
Literature Survey & Knowledge Graph Agent
------------------------------------------
Backend logic for IsoScreenAI Page 3.

Pipeline:
  1. Query PubMed (via Biopython's Bio.Entrez) for a target protein / gene / disease.
  2. Fetch abstracts for the top N hits.
  3. Send abstracts to Groq (Llama 3.3) with a strict JSON schema prompt to extract
     biomedical relationships (Protein-Disease, Drug-Target, Mechanism-of-Action, etc.)
  4. Return a structured payload: articles, relationships (nodes/edges), and an
     executive summary, ready for rendering in pages/3_Literature_Survey.py.

Mirrors the resilience patterns used in agents/protein_agent.py:
  - defensive parsing of LLM output (strip code fences, salvage partial JSON)
  - explicit handling of empty PubMed results
  - Groq errors (rate limit / token limit) surfaced as clean exceptions, not crashes
"""

import json
import logging
import re
import time
from typing import Optional

from Bio import Entrez
from groq import Groq

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# NCBI requires an identifying email for Entrez API courtesy/rate-limit purposes.
# Add ENTREZ_EMAIL (and optionally NCBI_API_KEY) to config.py.
Entrez.email = getattr(config, "ENTREZ_EMAIL", "isoscreenai@example.com")
_NCBI_API_KEY = getattr(config, "NCBI_API_KEY", None)
if _NCBI_API_KEY:
    Entrez.api_key = _NCBI_API_KEY

# Reuse the same Groq model/key convention as protein_agent.py.
_GROQ_API_KEY = getattr(config, "GROQ_API_KEY", None)
_GROQ_MODEL = getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile")

_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else Groq()


class LiteratureAgentError(Exception):
    """Raised for any recoverable failure in the literature survey pipeline."""


# ---------------------------------------------------------------------------
# Step 1: PubMed search
# ---------------------------------------------------------------------------

def search_pubmed(query: str, max_results: int = 8, retries: int = 2) -> list[str]:
    """Return a list of PMIDs for the given query."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:  # network hiccups, NCBI 429s, etc.
            last_err = e
            logger.warning("Entrez esearch failed (attempt %d): %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise LiteratureAgentError(f"PubMed search failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Step 2: Fetch abstracts
# ---------------------------------------------------------------------------

def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract + journal/year metadata for a list of PMIDs."""
    if not pmids:
        return []

    try:
        handle = Entrez.efetch(
            db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        raise LiteratureAgentError(f"PubMed fetch failed: {e}")

    articles = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]
            pmid = str(medline["PMID"])
            title = str(art.get("ArticleTitle", "")).strip()

            abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts).strip()

            year = None
            try:
                year = art["Journal"]["JournalIssue"]["PubDate"].get("Year")
            except Exception:
                pass

            journal = str(art.get("Journal", {}).get("Title", "")).strip()

            if title or abstract:
                articles.append(
                    {
                        "pmid": pmid,
                        "title": title or "(no title available)",
                        "abstract": abstract or "(no abstract available)",
                        "journal": journal,
                        "year": year,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }
                )
        except Exception as e:
            logger.warning("Skipping malformed PubMed record: %s", e)
            continue

    return articles


# ---------------------------------------------------------------------------
# Step 3: LLM relationship extraction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a biomedical knowledge-extraction engine used in a \
drug-discovery research tool. You read scientific abstracts and extract structured \
relationships suitable for a knowledge graph.

Return ONLY valid JSON (no markdown fences, no commentary) matching exactly this schema:

{
  "summary": "3-5 sentence executive synthesis across all provided abstracts",
  "nodes": [
    {"id": "string, short canonical entity name", "type": "Protein|Gene|Disease|Drug|Pathway|Mechanism|Other"}
  ],
  "edges": [
    {"source": "node id", "target": "node id", "relation": "short verb phrase, e.g. 'inhibits', 'associated_with', 'upregulates'", "pmid": "source PMID for this claim"}
  ]
}

Rules:
- Only extract relationships that are explicitly or strongly implicitly supported by the text.
- Reuse the exact same "id" string for the same entity across nodes/edges (canonicalize casing/synonyms).
- Prefer specific relation verbs (e.g. "inhibits", "biomarker_for", "upregulates", "target_of") over generic ones.
- Keep node ids short (protein/gene symbols, disease names, drug names).
- If abstracts contain no extractable relationships, return empty nodes/edges arrays but still write the summary.
- Do not invent PMIDs; only use the ones provided with each abstract.
"""


def _build_user_prompt(query: str, articles: list[dict]) -> str:
    blocks = []
    for a in articles:
        blocks.append(
            f"PMID: {a['pmid']}\nTitle: {a['title']}\nAbstract: {a['abstract']}"
        )
    joined = "\n\n---\n\n".join(blocks)
    return (
        f"Research query/target: {query}\n\n"
        f"Below are {len(articles)} PubMed abstracts. Extract the knowledge graph "
        f"as specified.\n\n{joined}"
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _salvage_json(text: str) -> Optional[dict]:
    """Best-effort recovery if the model wraps JSON in extra prose."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_relationships_with_llm(query: str, articles: list[dict]) -> dict:
    """Call Groq to extract a knowledge graph from the fetched abstracts.

    Truncates abstract text defensively to stay under Groq token limits, matching
    the guardrails already used for the protein report agent.
    """
    if not articles:
        return {"summary": "No abstracts were available to analyze.", "nodes": [], "edges": []}

    # Defensive truncation: cap total abstract characters sent to the LLM.
    MAX_CHARS_PER_ABSTRACT = 1800
    trimmed = [
        {**a, "abstract": a["abstract"][:MAX_CHARS_PER_ABSTRACT]} for a in articles
    ]

    user_prompt = _build_user_prompt(query, trimmed)

    try:
        response = _client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate_limit" in msg or "429" in msg:
            raise LiteratureAgentError(
                "Groq rate limit / token limit hit while extracting relationships. "
                "Try again shortly, or reduce the number of articles."
            )
        raise LiteratureAgentError(f"Groq relationship extraction failed: {e}")

    raw = response.choices[0].message.content
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = _salvage_json(cleaned)
        if data is None:
            logger.warning("Could not parse LLM JSON output; raw: %s", raw[:500])
            return {
                "summary": "The literature was retrieved, but relationship extraction "
                "returned malformed output. Try again or reduce the number of articles.",
                "nodes": [],
                "edges": [],
            }

    data.setdefault("summary", "")
    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    return data


# ---------------------------------------------------------------------------
# Step 4: Orchestration
# ---------------------------------------------------------------------------

def run_literature_survey(query: str, max_results: int = 8) -> dict:
    """Full pipeline entry point used by the Streamlit page.

    Returns:
        {
            "query": str,
            "articles": [ {pmid, title, abstract, journal, year, url}, ... ],
            "summary": str,
            "nodes": [ {id, type}, ... ],
            "edges": [ {source, target, relation, pmid}, ... ],
        }
    """
    query = (query or "").strip()
    if not query:
        raise LiteratureAgentError("Please enter a target protein, gene, or disease query.")

    pmids = search_pubmed(query, max_results=max_results)
    if not pmids:
        return {
            "query": query,
            "articles": [],
            "summary": f"No PubMed results found for '{query}'. Try a broader or "
            f"differently-worded query.",
            "nodes": [],
            "edges": [],
        }

    articles = fetch_abstracts(pmids)
    extraction = extract_relationships_with_llm(query, articles)

    return {
        "query": query,
        "articles": articles,
        "summary": extraction.get("summary", ""),
        "nodes": extraction.get("nodes", []),
        "edges": extraction.get("edges", []),
    }


# ---------------------------------------------------------------------------
# Helper: de-dupe / sanitize graph data before handing to the UI layer
# ---------------------------------------------------------------------------

def build_graph_data(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """De-duplicate nodes and drop edges that reference missing nodes.

    Keeps the UI layer (streamlit-agraph) simple and crash-free even if the
    LLM output has minor inconsistencies.
    """
    seen = {}
    clean_nodes = []
    for n in nodes:
        node_id = str(n.get("id", "")).strip()
        if not node_id or node_id in seen:
            continue
        seen[node_id] = True
        clean_nodes.append({"id": node_id, "type": n.get("type", "Other") or "Other"})

    valid_ids = set(seen.keys())
    clean_edges = []
    edge_seen = set()
    for e in edges:
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if not src or not tgt or src not in valid_ids or tgt not in valid_ids:
            continue
        key = (src, tgt, e.get("relation", ""))
        if key in edge_seen:
            continue
        edge_seen.add(key)
        clean_edges.append(
            {
                "source": src,
                "target": tgt,
                "relation": e.get("relation", "related_to") or "related_to",
                "pmid": e.get("pmid", ""),
            }
        )

    return clean_nodes, clean_edges
