import streamlit as st
import os
import config
from agent import graph

st.set_page_config(
    page_title="GraphDrugPred v1",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 GraphDrugPred (v1)")
st.caption("Modular Screening System: Sequence-Only Deep Embedding Target Classifier")

# Configuration Interface Sidebar
st.sidebar.header("Pipeline Configuration")
user_api_key = st.sidebar.text_input("Google AI Studio API Key", type="password")

if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Target Model Engine**: `{config.MODEL_NAME}`")
st.sidebar.markdown(f"**Embedding Space Vector**: `{config.EMBEDDING_DIM} Dimensions`")
st.sidebar.markdown(f"**System Threshold**: `{config.DRUGGABILITY_THRESHOLD}`")

# Central Sequence Input Form
st.markdown("### 1. Target Sequence Specifications")
input_text = st.text_area(
    "Insert Target Sequence (FASTA Format):",
    placeholder=">Sample_Protein\nMKKVLVINGFGRIIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVING...",
    height=180
)

uploaded_file = st.file_uploader("Or select standard .fasta or .fa file profiles", type=["fasta", "fa"])
if uploaded_file is not None:
    input_text = uploaded_file.read().decode("utf-8")

if st.button("Execute Pipeline Diagnostics", type="primary"):
    if not input_text.strip():
        st.error("Execution Blocked: Please submit a valid input sequence payload.")
    else:
        # Define execution payload
        execution_payload = {
            "fasta_content": input_text,
            "sequence": "",
            "embedding": None,
            "prediction": None,
            "confidence": None,
            "report": "",
            "errors": []
        }
        
        with st.status("Initializing Local Multi-Node Workflow Agent...", expanded=True) as state_monitor:
            state_monitor.write("Running step: Sanitizing input characters...")
            runtime_output = graph.invoke(execution_payload)
            
            if runtime_output.get("errors") and not runtime_output.get("prediction"):
                state_monitor.update(label="Diagnostic Workflow Fault Encountered", state="error")
                for fault in runtime_output["errors"]:
                    st.error(fault)
            else:
                state_monitor.update(label="Graph Sequence Orchestration Finished Successfully", state="complete")
        
        # Display Dashboard Analytics upon pipeline completion
        if runtime_output.get("prediction") is not None:
            st.markdown("### 2. Screening Metrics Dashboard")
            metric_cols = st.columns(3)
            
            raw_prob = runtime_output["prediction"]
            is_druggable = raw_prob >= config.DRUGGABILITY_THRESHOLD
            verdict_text = "Highly Druggable" if is_druggable else "Poorly Druggable"
            
            with metric_cols[0]:
                st.metric(
                    label="Druggability Verdict",
                    value=verdict_text,
                    delta="Positive Profile" if is_druggable else "Negative Profile",
                    delta_color="normal" if is_druggable else "inverse"
                )
            with metric_cols[1]:
                st.metric(label="Target Probability Score", value=f"{raw_prob:.4f}")
            with metric_cols[2]:
                st.metric(label="Inference Confidence Metric", value=f"{runtime_output['confidence']:.4f}")
                
            st.markdown("---")
            st.markdown("### 3. Analytical Synthesis Report")
            st.markdown(runtime_output["report"])
            
            if runtime_output.get("errors"):
                with st.expander("System Runtime Logs & Discovered Warnings"):
                    for warning in runtime_output["errors"]:
                        st.warning(warning)
