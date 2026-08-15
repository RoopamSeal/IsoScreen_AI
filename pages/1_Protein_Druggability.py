import streamlit as st
import pandas as pd
from agents.protein_agent import analyze_protein_sequence, predict_gene_ontology_terms, generate_groq_report

st.set_page_config(page_title="Protein Druggability & Function Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Protein Druggability & Function Predictor")
st.markdown("Evaluate target proteins using **ESM-2 embeddings**, physicochemical properties, **Gene Ontology (GO) functional prediction**, and **Groq AI reporting**.")

# Sidebar for API Key
st.sidebar.header("🔑 AI Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", help="Required for GO term prediction and AI reports.")

# Input area
seq_input = st.text_area(
    "Protein Sequence (FASTA format or Raw Amino Acids)",
    height=150
)

if st.button("Run Pipeline"):
    if not seq_input.strip():
        st.error("Please enter a valid protein sequence.")
    else:
        with st.spinner("Extracting ESM-2 embeddings and computing sequence features..."):
            try:
                metrics = analyze_protein_sequence(seq_input)
                st.success("Sequence analysis complete!")
                
                # Display Metrics in Columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sequence Length", f"{metrics['length']} aa")
                col2.metric("Molecular Weight", f"{metrics['mol_weight']} Da")
                col3.metric("GRAVY Index", f"{metrics['gravy']}")
                col4.metric("Instability Index", f"{metrics['instability_index']}")
                
                # Tabs for different output categories
                tab1, tab2, tab3 = st.tabs(["🎯 Druggability & Metrics", "🧬 Functional GO Prediction", "📑 AI Research Report"])
                
                with tab1:
                    st.subheader("Druggability Assessment")
                    score = metrics['druggability_score']
                    st.progress(score)
                    st.write(f"**Estimated Druggability Confidence Score:** `{score * 100:.1f}%`")
                    
                    if score > 0.7:
                        st.markdown("🟢 **High Promising Target:** Favorable profile for pocket formation and stability.")
                    else:
                        st.markdown("🟡 **Moderate Target:** May require structural refinement or domain filtering.")
                        
                    st.markdown("---")
                    st.subheader("Amino Acid Composition (%)")
                    df_aa = pd.DataFrame(list(metrics['aa_composition'].items()), columns=["Amino Acid", "Percentage (%)"])
                    st.bar_chart(df_aa.set_index("Amino Acid"))
                
                with tab2:
                    st.subheader("Gene Ontology (GO) Functional Classification")
                    st.markdown("Multi-label prediction of molecular functions, biological processes, and cellular components derived from sequence embeddings.")
                    
                    if not groq_api_key:
                        st.warning("⚠️ Enter your Groq API key in the sidebar to generate functional GO predictions.")
                    else:
                        with st.spinner("Predicting GO terms using model features..."):
                            go_prediction = predict_gene_ontology_terms(metrics, groq_api_key)
                            st.markdown(go_prediction)
                
                with tab3:
                    st.subheader("Executive Research Report")
                    if not groq_api_key:
                        st.warning("⚠️ Enter your Groq API key in the sidebar to unlock the AI research report.")
                    else:
                        with st.spinner("Generating expert report with Groq..."):
                            report = generate_groq_report(metrics, groq_api_key)
                            st.markdown(report)
                            
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
