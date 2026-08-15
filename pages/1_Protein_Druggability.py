import streamlit as st
from agents.protein_agent import analyze_protein_sequence, generate_groq_report

st.set_page_config(page_title="Protein Druggability Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Protein Druggability Predictor")
st.markdown("Evaluate target protein sequences using **ESM-2 deep learning embeddings**, physicochemical descriptors, and **Groq AI reporting**.")

# Input area
seq_input = st.text_area(
    "Protein Sequence (FASTA / Raw Amino Acids)",
    value="MKTAYIAKQRQISFVKSHFSRQLEERGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALP",
    height=150
)

if st.button("Run Druggability Pipeline"):
    if not seq_input.strip():
        st.error("Please enter a valid protein sequence.")
    else:
        # Retrieve secrets safely
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        hf_token = st.secrets.get("HUGGINGFACE_API_KEY", None)
        
        if not groq_key:
            st.error("❌ `GROQ_API_KEY` is missing from Streamlit secrets. Please configure it in `.streamlit/secrets.toml`.")
        else:
            with st.spinner("Running ESM-2 sequence analysis and calculating metrics..."):
                try:
                    # Call backend agent logic with HF token if available
                    metrics = analyze_protein_sequence(seq_input, hf_token=hf_token)
                    
                    st.success("Analysis complete!")
                    
                    # Display Metrics in Columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Sequence Length", f"{metrics['length']} aa")
                    col2.metric("Molecular Weight", f"{metrics['mol_weight']} Da")
                    col3.metric("GRAVY Index", f"{metrics['gravy']}")
                    col4.metric("Instability Index", f"{metrics['instability_index']}")
                    
                    st.markdown("---")
                    st.subheader("📊 Druggability Assessment")
                    
                    score = metrics['druggability_score']
                    st.progress(score)
                    st.write(f"**Estimated Druggability Confidence Score:** `{score * 100:.1f}%`")
                    
                    if score > 0.7:
                        st.markdown("🟢 **High Promising Target:** Favorable profile for pocket formation and stability.")
                    else:
                        st.markdown("🟡 **Moderate Target:** May require structural refinement or domain filtering.")
                    
                    # Groq Report Section
                    st.markdown("---")
                    st.subheader("📑 Groq AI Research Report")
                    
                    with st.spinner("Generating expert drug discovery report with Groq (Llama 3.3)..."):
                        report = generate_groq_report(metrics, groq_key)
                        st.markdown(report)
                            
                except Exception as e:
                    st.error(f"An error occurred during pipeline execution: {e}")
