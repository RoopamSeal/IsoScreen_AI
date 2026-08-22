"""
Page 3 - Literature Survey Agent
UI layer only. All pipeline logic lives in agents/literature_agent.py.

Design: each pipeline stage is invoked and displayed separately (rather than
one opaque "run everything" button), matching the six-step review process.
Expensive LLM stages (screening, extraction, synthesis) are cached in
st.session_state and only re-run on an explicit user action; the relevance
threshold slider re-filters already-screened data locally (free, instant).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from agents.literature_agent import (
    LiteratureAgentError,
    build_markdown_report,
    dedupe_articles,
    extract_paper_details,
    fetch_articles,
    screen_abstracts,
    search_pubmed,
    select_included,
    synthesize_review,
    to_records,
)

st.set_page_config(page_title="Literature Survey | IsoScreenAI", page_icon="📚", layout="wide")

st.title("📚 Literature Survey Agent")
st.caption(
    "Enter a research question to search PubMed, screen and deduplicate results, "
    "extract structured study details, surface gaps and conflicts in the current "
    "literature, and download a complete review report."
)

for key in ["fetched", "removed", "screened", "extracted", "synthesis", "report_md", "research_question", "included"]:
    st.session_state.setdefault(key, None)

# ---------------------------------------------------------------------------
# Step 1 & 2: Query -> Search & Fetch
# ---------------------------------------------------------------------------

with st.form("search_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        research_question = st.text_input(
            "Research question",
            placeholder="e.g. Does PCSK9 inhibition reduce cardiovascular events in statin-intolerant patients?",
        )
    with col2:
        max_results = st.slider("Max articles to fetch", min_value=5, max_value=30, value=15)
    search_submitted = st.form_submit_button("1-2. Search & Fetch Articles", type="primary")

if search_submitted:
    if not research_question.strip():
        st.warning("Enter a research question first.")
        st.stop()
    with st.spinner("Searching PubMed and fetching articles..."):
        try:
            pmids = search_pubmed(research_question, max_results=max_results)
            if not pmids:
                st.warning(f"No PubMed results found for '{research_question}'. Try rephrasing.")
                st.stop()
            fetched = fetch_articles(pmids)
            deduped, removed = dedupe_articles(fetched)
        except LiteratureAgentError as e:
            st.error(str(e))
            st.stop()

    # New search invalidates all downstream cached stages.
    st.session_state["research_question"] = research_question
    st.session_state["fetched"] = deduped
    st.session_state["removed"] = removed
    st.session_state["screened"] = None
    st.session_state["extracted"] = None
    st.session_state["synthesis"] = None
    st.session_state["report_md"] = None
    st.session_state["included"] = None

# ---------------------------------------------------------------------------
# Step 2/3 display: fetched + deduplicated articles
# ---------------------------------------------------------------------------

if st.session_state["fetched"]:
    fetched = st.session_state["fetched"]
    removed = st.session_state["removed"]

    st.divider()
    st.subheader("2-3. Retrieved Articles (deduplicated)")
    st.caption(f"{len(fetched)} unique articles retained — {removed} duplicate(s) removed.")

    df_fetched = pd.DataFrame(to_records(fetched))[["title", "year", "journal", "url"]]
    df_fetched.columns = ["Title", "Year", "Journal", "Link"]
    st.dataframe(
        df_fetched,
        use_container_width=True,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open ↗")},
    )

    if st.button("3. Screen Abstracts for Relevance"):
        progress_bar = st.progress(0.0, text="Starting screening...")

        def _on_progress(msg, frac):
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            screened = screen_abstracts(
                st.session_state["research_question"], fetched, progress_callback=_on_progress
            )
        except LiteratureAgentError as e:
            st.error(str(e))
            st.stop()
        progress_bar.empty()
        st.session_state["screened"] = screened
        st.session_state["extracted"] = None
        st.session_state["synthesis"] = None
        st.session_state["report_md"] = None
        st.session_state["included"] = None

# ---------------------------------------------------------------------------
# Step 3 display: screening results + relevance threshold (free, local filter)
# ---------------------------------------------------------------------------

if st.session_state["screened"]:
    screened = st.session_state["screened"]

    st.divider()
    st.subheader("3. Abstract Screening")

    fig_col, table_col = st.columns([1, 2])
    with fig_col:
        score_df = pd.DataFrame({"relevance_score": [s.relevance_score for s in screened]})
        st.caption("Relevance score distribution")
        st.bar_chart(score_df["relevance_score"].value_counts().sort_index())

    with table_col:
        threshold = st.slider("Relevance threshold (include if score ≥)", 0, 10, 6)
        max_papers = st.slider("Max papers to carry forward to extraction", 3, 20, 12)
        included = select_included(screened, relevance_threshold=threshold, max_papers=max_papers)
        st.caption(f"{len(included)} of {len(screened)} articles selected as high-relevance.")

    df_screened = pd.DataFrame(to_records(screened))[
        ["title", "year", "relevance_score", "recommended_include", "screening_reason", "url"]
    ]
    df_screened.columns = ["Title", "Year", "Relevance", "LLM Include?", "Reason", "Link"]
    df_screened = df_screened.sort_values("Relevance", ascending=False)
    st.dataframe(
        df_screened,
        use_container_width=True,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open ↗")},
    )

    st.session_state["included"] = included

    if included and st.button("4-5. Extract Details & Synthesize Review", type="primary"):
        progress_bar = st.progress(0.0, text="Starting extraction...")

        def _on_progress(msg, frac):
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            extracted = extract_paper_details(included, progress_callback=_on_progress)
            progress_bar.progress(0.9, text="Synthesizing gaps and conflicts...")
            synthesis = synthesize_review(st.session_state["research_question"], extracted)
        except LiteratureAgentError as e:
            st.error(str(e))
            st.stop()
        progress_bar.empty()

        st.session_state["extracted"] = extracted
        st.session_state["synthesis"] = synthesis
        st.session_state["report_md"] = build_markdown_report(
            st.session_state["research_question"],
            st.session_state["fetched"],
            st.session_state["removed"],
            screened,
            included,
            extracted,
            synthesis,
        )
    elif not included:
        st.info("No articles currently meet the relevance threshold — lower it to proceed.")

# ---------------------------------------------------------------------------
# Step 4 display: extracted study details
# ---------------------------------------------------------------------------

if st.session_state["extracted"]:
    extracted = st.session_state["extracted"]

    st.divider()
    st.subheader("4. Extracted Study Details")

    df_overview = pd.DataFrame(to_records(extracted))[["title", "year", "study_type", "sample_size"]]
    df_overview.columns = ["Title", "Year", "Study Type", "Sample Size"]
    st.dataframe(df_overview, use_container_width=True)

    type_counts = df_overview["Study Type"].value_counts()
    st.caption("Study type breakdown")
    st.bar_chart(type_counts)

    for p in extracted:
        with st.expander(f"{p.title} ({p.year or 'n.d.'})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Study type:** {p.study_type}")
                st.markdown(f"**Methods:** {p.methods}")
                st.markdown(f"**Datasets:** {p.datasets}")
            with c2:
                st.markdown(f"**Key findings:** {p.key_findings}")
                st.markdown(f"**Sample size:** {p.sample_size}")
                st.markdown(f"**Limitations:** {p.limitations}")

# ---------------------------------------------------------------------------
# Step 5 display: synthesis (gaps, conflicts, open questions)
# ---------------------------------------------------------------------------

if st.session_state["synthesis"]:
    synthesis = st.session_state["synthesis"]

    st.divider()
    st.subheader("5. Synthesis: Gaps, Conflicts & Open Questions")
    st.write(synthesis.overall_synthesis)

    g_col, c_col, q_col = st.columns(3)
    with g_col:
        st.markdown("**🕳️ Gaps in the Literature**")
        if synthesis.gaps:
            for g in synthesis.gaps:
                st.markdown(f"- {g}")
        else:
            st.caption("None identified.")
    with c_col:
        st.markdown("**⚖️ Conflicting Findings**")
        if synthesis.conflicts:
            for c in synthesis.conflicts:
                st.markdown(f"- **{c.topic}:** {c.description}")
                st.caption(f"PMIDs: {', '.join(c.pmids) if c.pmids else 'n/a'}")
        else:
            st.caption("None identified.")
    with q_col:
        st.markdown("**❓ Open Questions**")
        if synthesis.open_questions:
            for q in synthesis.open_questions:
                st.markdown(f"- {q}")
        else:
            st.caption("None identified.")

# ---------------------------------------------------------------------------
# Step 6: Downloadable report
# ---------------------------------------------------------------------------

if st.session_state["report_md"]:
    st.divider()
    st.subheader("6. Download Full Report")
    st.download_button(
        "📥 Download Literature Review (Markdown)",
        data=st.session_state["report_md"],
        file_name=f"literature_review_{st.session_state['research_question'][:40].replace(' ', '_')}.md",
        mime="text/markdown",
        type="primary",
    )
    with st.expander("Preview report"):
        st.markdown(st.session_state["report_md"])

if not st.session_state["fetched"]:
    st.info("Enter a research question above to begin.")
