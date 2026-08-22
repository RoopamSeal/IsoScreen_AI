"""
Page 3 - Literature Survey Agent
UI layer only; pipeline logic lives in agents/literature_agent.py.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from agents.literature_agent import LiteratureAgentError, run_literature_survey

st.set_page_config(page_title="Literature Survey | IsoScreenAI", page_icon="📚", layout="wide")

st.title("📚 Literature Survey Agent")
st.caption(
    "Search PubMed for a research question, rank papers by relevance, and generate "
    "a downloadable literature review report."
)

with st.form("survey_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Research question",
            placeholder="e.g. Does PCSK9 inhibition reduce cardiovascular events in statin-intolerant patients?",
        )
    with col2:
        max_results = st.slider("Max articles", 5, 30, 15)
    submitted = st.form_submit_button("Run Literature Survey", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a research question first.")
        st.stop()
    with st.spinner("Searching PubMed, ranking papers, and writing the report..."):
        try:
            result = run_literature_survey(query, max_results=max_results)
        except LiteratureAgentError as e:
            st.error(str(e))
            st.stop()
    st.session_state["lit_result"] = result

result = st.session_state.get("lit_result")

if result:
    st.divider()
    st.subheader("Papers Ranked by Relevance")
    df = pd.DataFrame(result["articles"])[["title", "year", "relevance_score", "url"]]
    df.columns = ["Title", "Year", "Relevance", "Link"]
    st.dataframe(
        df,
        use_container_width=True,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open ↗")},
    )

    st.divider()
    st.subheader("Literature Review Report")
    st.download_button(
        "📥 Download Report (Markdown)",
        data=result["report_md"],
        file_name=f"literature_review_{result['query'][:40].replace(' ', '_')}.md",
        mime="text/markdown",
        type="primary",
    )
    st.markdown(result["report_md"])
else:
    st.info("Enter a research question above to begin.")
