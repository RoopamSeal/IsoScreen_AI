import streamlit as st
import config

st.set_page_config(
    page_title="GraphDrugPred v1",
    page_icon="🧬",
    layout="wide"
)

# Safety check for Groq Key
try:
    _ = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Security Halt: GROQ_API_KEY is missing from your Streamlit Secrets.")
    st.stop()

# Import the graph after passing the security check
from agent import graph

st.title("🧬 GraphDrugPred (v1)")
st.caption("Modular Screening System: Sequence-Only Deep Embedding Target Classifier")

# Configuration Interface Sidebar
st.sidebar.header("Pipeline Configuration")
st.sidebar.markdown(f"**Target Model Engine**: `{config.MODEL_NAME}`")
st.sidebar.markdown(f"**Embedding Space Vector**: `{config.EMBEDDING_DIM} Dimensions`")
st.sidebar.markdown(f"**System Threshold**: `{config.DRUGGABILITY_THRESHOLD}`")
st.sidebar.markdown("---")
st.sidebar.success("Status: Authenticated via Groq LPUs")

# Central Sequence Input Form
st.markdown("### 1. Target Sequence Specifications")
input_text = st.text_area(
    "Insert Target Sequence (FASTA Format):",
    placeholder=">Sample_Protein\nMKKVLVINGFGRIIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVING...",
    height=180
)

if st.button("Execute Pipeline Diagnostics", type="primary"):
    if not input_text.strip():
        st.error("Execution Blocked: Please submit a valid input sequence payload.")
    else:
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
