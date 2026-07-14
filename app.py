"""
===========================================================
GraphDrugPred / IsoScreenAI
Streamlit Application Frontend
===========================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import config
from predictor import get_predictor
from agent import create_graph

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title=f"{config.APP_NAME} v{config.VERSION}",
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
)

# ==========================================================
# CACHED GRAPH INITIALIZATION
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_workflow_graph():
    """
    Safely compiles the LangGraph workflow once per server boot,
    injecting the cached PyTorch predictor to prevent RAM limits from exceeding.
    """
    predictor = get_predictor()
    return create_graph(predictor)

graph = load_workflow_graph()

# ==========================================================
# HEADER & INPUT
# ==========================================================

st.title(f"{config.APP_ICON} {config.APP_NAME}")
st.caption("Sequence-based Protein Druggability Prediction using ESM-2 + LangGraph")

st.header("1. Target Protein Sequence")

input_mode = st.radio(
    "Choose input method:",
    ("Paste FASTA", "Upload FASTA File"),
    horizontal=True
)

fasta_text = ""

if input_mode == "Paste FASTA":
    fasta_text = st.text_area(
        "Paste sequence below:",
        height=200,
        placeholder=">Example_Target\nMKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE...",
    )
else:
    uploaded = st.file_uploader(
        "Upload sequence file:",
        type=["fasta", "fa", "faa", "txt"],
    )
    if uploaded:
        fasta_text = uploaded.read().decode("utf-8")

# ==========================================================
# EXECUTION PIPELINE
# ==========================================================

if st.button("Execute Pipeline Diagnostics", type="primary", use_container_width=True):
    if not fasta_text.strip():
        st.error("Execution Blocked: Please submit a valid FASTA sequence payload.")
        st.stop()

    initial_state = {
        "fasta_content": fasta_text,
        "errors": [],
    }

    progress_bar = st.progress(0)

    with st.spinner("Orchestrating AI agents & running structural simulations..."):
        progress_bar.progress(25)
        result = graph.invoke(initial_state)
        progress_bar.progress(100)

    progress_bar.empty()

    # ------------------------------------------------------
    # ERROR & WARNING HANDLING
    # ------------------------------------------------------
    if result.get("errors"):
        st.warning("Pipeline completed with diagnostic warnings:")
        for err in result["errors"]:
            st.error(err)

    if result.get("prediction") is None:
        st.stop()

    # ======================================================
    # METRICS DASHBOARD
    # ======================================================
    st.success("Target Assessment Pipeline Complete")
    st.divider()

    st.header("2. Biophysical Diagnostic Matrix")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label="Druggability Probability",
        value=f"{result['prediction']:.3f}",
        help="Sequence-derived probability calculated by the ESM-2 transformer classification layer."
    )
    c2.metric(
        label="Model Confidence",
        value=f"{result['confidence']:.3f}",
        help="Distance from classification threshold. Higher values signify stronger mathematical assurance."
    )
    c3.metric(
        label="Mean pLDDT",
        value=f"{result['mean_plddt']:.1f}",
        help="Per-residue backbone structural confidence. Scores >70 indicate stable, well-folded structures."
    )
    c4.metric(
        label="Binding Free Energy (ΔG)",
        value=f"{result['binding_free_energy']:.2f} kcal/mol",
        help="Thermodynamic binding affinity from ligand docking sweeps. Lower (more negative) values indicate stronger binding."
    )

    # ======================================================
    # VISUALIZATIONS
    # ======================================================
    st.header("3. Multi-Dimensional Visualizations")
    left, right = st.columns(2)

    # ------------------------------------------------------
    # Gauge Chart
    # ------------------------------------------------------
    with left:
        if config.ENABLE_GAUGE_CHART:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["prediction"],
                    title={"text": "Target Druggability Score"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "royalblue"},
                        "threshold": {
                            "line": {"color": "crimson", "width": 4},
                            "value": config.DRUGGABILITY_THRESHOLD,
                        },
                        "steps": [
                            {"range": [0, 0.5], "color": "lightgray"},
                            {"range": [0.5, 0.7], "color": "darkgray"},
                        ]
                    },
                )
            )
            gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge, use_container_width=True)

    # ------------------------------------------------------
    # Radar Chart
    # ------------------------------------------------------
    with right:
        if config.ENABLE_RADAR_CHART:
            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=[
                        result["evolutionary_conservation"] * 100,
                        result["mean_plddt"],
                        abs(result["binding_free_energy"]) * 10,
                        100 - result["predicted_disorder"],
                    ],
                    theta=["Conservation", "pLDDT Score", "Binding Affinity", "Folding Order"],
                    fill="toself",
                    marker=dict(color="darkblue")
                )
            )
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(radar, use_container_width=True)

    # ======================================================
    # PAE HEATMAP
    # ======================================================
    if config.ENABLE_PAE_VISUALIZATION and result.get("pae_matrix"):
        st.subheader("Predicted Alignment Error (PAE) Matrix")
        fig_pae = px.imshow(
            result["pae_matrix"],
            color_continuous_scale="Viridis",
            labels=dict(x="Scored Residue", y="Aligned Residue", color="Expected Error (Å)")
        )
        fig_pae.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pae, use_container_width=True)

    # ======================================================
    # SEQUENCE DISPLAY
    # ======================================================
    st.subheader("Validated Target Residues")
    st.code(result["sequence"], language="text")

    # ======================================================
    # REPORT & DOWNLOAD
    # ======================================================
    st.header("4. AI Scientific Assessment Report")
    st.markdown(result["report"])

    st.download_button(
        label="📥 Download Academic Report (.md)",
        data=result["report"],
        file_name="IsoScreenAI_Target_Report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ======================================================
    # TECHNICAL METADATA
    # ======================================================
    with st.expander("🔬 Model Metadata & System Execution Parameters"):
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"**Engine Architecture:** `{config.MODEL_NAME}`")
        m2.markdown(f"**Embedding Vector:** `{config.EMBEDDING_DIM}D Space`")
        m3.markdown(f"**Decision Threshold:** `{config.DRUGGABILITY_THRESHOLD}`")
        st.markdown(f"**Processed Sequence Length:** `{len(result['sequence'])} amino acids`")
