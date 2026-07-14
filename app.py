import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from agent import graph
import config

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="IsoScreenAI",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 IsoScreenAI")
st.caption(
    "Sequence-based Protein Druggability Prediction using ESM-2 + LangGraph"
)

# ==========================================================
# INPUT
# ==========================================================

st.header("Protein Sequence")

input_mode = st.radio(
    "Choose input method",
    (
        "Paste FASTA",
        "Upload FASTA File",
    ),
)

fasta_text = ""

if input_mode == "Paste FASTA":

    fasta_text = st.text_area(
        "Paste FASTA",
        height=250,
        placeholder=""">Example
MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE""",
    )

else:

    uploaded = st.file_uploader(
        "Upload FASTA",
        type=["fasta", "fa", "faa", "txt"],
    )

    if uploaded:

        fasta_text = uploaded.read().decode("utf-8")

# ==========================================================
# RUN BUTTON
# ==========================================================

if st.button(
    "Analyze Protein",
    type="primary",
    use_container_width=True,
):

    if fasta_text.strip() == "":

        st.error("Please provide a FASTA sequence.")

        st.stop()

    initial_state = {
        "fasta_content": fasta_text,
        "errors": [],
    }

    progress = st.progress(0)

    with st.spinner("Running AI workflow..."):

        progress.progress(20)

        result = graph.invoke(initial_state)

        progress.progress(100)

    progress.empty()

    # ------------------------------------------------------
    # ERRORS
    # ------------------------------------------------------

    if result.get("errors"):

        st.error("Pipeline completed with warnings.")

        for err in result["errors"]:
            st.warning(err)

    if result.get("prediction") is None:

        st.stop()

    # ======================================================
    # RESULTS
    # ======================================================

    st.success("Analysis Complete")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Druggability",
        f"{result['prediction']:.3f}",
    )

    c2.metric(
        "Confidence",
        f"{result['confidence']:.3f}",
    )

    c3.metric(
        "Mean pLDDT",
        f"{result['mean_plddt']:.1f}",
    )

    c4.metric(
        "Binding ΔG",
        f"{result['binding_free_energy']:.2f}",
    )

    # ======================================================
    # VISUALIZATIONS
    # ======================================================

    st.header("Visualizations")

    left, right = st.columns(2)

    # ------------------------------------------------------
    # Gauge
    # ------------------------------------------------------

    with left:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["prediction"],
                title={"text": "Druggability Score"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "value": config.DRUGGABILITY_THRESHOLD,
                    },
                },
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

    # ------------------------------------------------------
    # Radar
    # ------------------------------------------------------

    with right:

        radar = go.Figure()

        radar.add_trace(
            go.Scatterpolar(
                r=[
                    result["evolutionary_conservation"] * 100,
                    result["mean_plddt"],
                    abs(result["binding_free_energy"]) * 10,
                    100 - result["predicted_disorder"],
                ],
                theta=[
                    "Conservation",
                    "pLDDT",
                    "Affinity",
                    "Order",
                ],
                fill="toself",
            )
        )

        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                )
            ),
            showlegend=False,
        )

        st.plotly_chart(
            radar,
            use_container_width=True,
        )

    # ======================================================
    # PAE
    # ======================================================

    st.subheader("Predicted Alignment Error")

    fig = px.imshow(
        result["pae_matrix"],
        color_continuous_scale="Viridis",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ======================================================
    # SEQUENCE
    # ======================================================

    st.subheader("Validated Protein Sequence")

    st.code(
        result["sequence"],
        language="text",
    )

    # ======================================================
    # REPORT
    # ======================================================

    st.header("AI Scientific Report")

    st.markdown(result["report"])

    # ======================================================
    # DOWNLOAD
    # ======================================================

    st.download_button(
        "Download Report",
        data=result["report"],
        file_name="IsoScreenAI_Report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ======================================================
    # TECHNICAL DETAILS
    # ======================================================

    with st.expander("Technical Details"):

        st.write(f"Model : {config.MODEL_NAME}")

        st.write(f"Embedding Size : {config.EMBEDDING_DIM}")

        st.write(f"Threshold : {config.DRUGGABILITY_THRESHOLD}")

        st.write(f"Sequence Length : {len(result['sequence'])}")
