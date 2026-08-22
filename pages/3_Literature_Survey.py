"""
Page 3 - Literature Survey & Knowledge Graph Agent
UI layer only. All heavy lifting lives in agents/literature_agent.py.
"""

import sys
import os

# Ensure project root is importable when Streamlit runs pages/ directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from agents.literature_agent import (
    LiteratureAgentError,
    build_graph_data,
    run_literature_survey,
)

try:
    from streamlit_agraph import Config, Edge, Node, agraph

    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False

st.set_page_config(page_title="Literature Survey | IsoScreenAI", page_icon="📚", layout="wide")

st.title("📚 Literature Survey & Knowledge Graph Agent")
st.caption(
    "Search PubMed for a target protein, gene, or disease and extract a structured "
    "knowledge graph of biomedical relationships (Protein-Disease, Drug-Target, "
    "Mechanism of Action) to accelerate early-stage target validation."
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

with st.form("literature_survey_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Target protein, gene, or disease",
            placeholder="e.g. KRAS G12C, EGFR non-small cell lung cancer, PCSK9",
        )
    with col2:
        max_results = st.slider("Max articles", min_value=3, max_value=20, value=8)
    submitted = st.form_submit_button("Run Literature Survey", type="primary")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if submitted:
    if not query.strip():
        st.warning("Enter a query before running the survey.")
        st.stop()

    with st.spinner(f"Searching PubMed and extracting relationships for '{query}'..."):
        try:
            result = run_literature_survey(query, max_results=max_results)
        except LiteratureAgentError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error during literature survey: {e}")
            st.stop()

    st.session_state["lit_survey_result"] = result

# ---------------------------------------------------------------------------
# Display (persisted in session_state so widget interactions don't re-run the pipeline)
# ---------------------------------------------------------------------------

result = st.session_state.get("lit_survey_result")

if result:
    st.divider()
    st.subheader("Executive Summary")
    st.write(result["summary"] or "_No summary generated._")

    articles = result["articles"]
    nodes_raw, edges_raw = result["nodes"], result["edges"]
    nodes, edges = build_graph_data(nodes_raw, edges_raw)

    tab_graph, tab_articles, tab_relationships = st.tabs(
        ["🕸️ Knowledge Graph", "📄 Articles", "🔗 Relationships Table"]
    )

    # -- Knowledge graph tab -------------------------------------------------
    with tab_graph:
        if not nodes:
            st.info("No relationships were extracted to visualize. Try a more specific query.")
        elif AGRAPH_AVAILABLE:
            type_colors = {
                "Protein": "#4C9AFF",
                "Gene": "#57D9A3",
                "Disease": "#FF7452",
                "Drug": "#FFAB00",
                "Pathway": "#998DD9",
                "Mechanism": "#79E2F2",
                "Other": "#B3BAC5",
            }
            agraph_nodes = [
                Node(
                    id=n["id"],
                    label=n["id"],
                    size=22,
                    color=type_colors.get(n["type"], type_colors["Other"]),
                )
                for n in nodes
            ]
            agraph_edges = [
                Edge(source=e["source"], target=e["target"], label=e["relation"])
                for e in edges
            ]
            config = Config(
                width=1000,
                height=550,
                directed=True,
                physics=True,
                hierarchical=False,
                collapsible=False,
            )
            agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

            legend_cols = st.columns(len(type_colors))
            for col, (t, c) in zip(legend_cols, type_colors.items()):
                col.markdown(
                    f"<span style='color:{c}'>●</span> {t}", unsafe_allow_html=True
                )
        else:
            st.warning(
                "Install `streamlit-agraph` for an interactive graph "
                "(`pip install streamlit-agraph`). Showing relationships as a table instead."
            )
            st.dataframe(pd.DataFrame(edges), use_container_width=True)

    # -- Articles tab ----------------------------------------------------------
    with tab_articles:
        if not articles:
            st.info("No PubMed articles found for this query.")
        for a in articles:
            with st.expander(f"{a['title']}  ({a.get('year', 'n.d.')})"):
                st.markdown(f"**Journal:** {a.get('journal', 'N/A')}")
                st.markdown(f"**PMID:** [{a['pmid']}]({a['url']})")
                st.write(a["abstract"])

    # -- Relationships table tab ------------------------------------------------
    with tab_relationships:
        if edges:
            df = pd.DataFrame(edges)[["source", "relation", "target", "pmid"]]
            df.columns = ["Source", "Relation", "Target", "PMID"]
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download relationships as CSV",
                data=df.to_csv(index=False),
                file_name=f"literature_relationships_{result['query'].replace(' ', '_')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No relationships were extracted.")
else:
    st.info("Enter a query above and run the survey to see results.")
