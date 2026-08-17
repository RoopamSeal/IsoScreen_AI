import streamlit as st
import pandas as pd
from agents.protein_agent import analyze_protein_sequence, predict_gene_ontology_terms, generate_groq_report

st.set_page_config(page_title="Protein Druggability & Function Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Protein Druggability & Function Predictor")
st.markdown("Evaluate target proteins using **ESM-2 embeddings**, physicochemical properties, **Gene Ontology (GO) functional prediction**, and **Groq AI reporting**.")

# 1. Input Requirements Parameter Guide
with st.expander("📌 Input Data Requirements & Guidelines", expanded=False):
    st.markdown(
        """
        * **Accepted Format**: **FASTA format** (with an optional header line starting with `>`) or **Raw Amino Acid Sequence** (single-letter standard IUPAC amino acid codes: `A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`).
        * **Recommended Length**: 50 to 1,500 amino acids for optimal feature extraction and embedding performance.
        * **Example Input**:
          ```text
          >Target_Protein_1
          MKTAYIAKQRQISFVKSHFSRQLEERGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALP
          ```
        """
    )

# 2. Step-by-Step Workflow Layout
with st.expander("⚙️ Pipeline Workflow Overview", expanded=False):
    st.markdown(
        """
        1. **Sequence Ingestion & Validation**: Strips FASTA headers and verifies valid single-letter amino acid residues.
        2. **BioPython Feature Extraction**: Computes physicochemical descriptors (Molecular Weight, GRAVY, Instability Index, Aromaticity, and Amino Acid Composition).
        3. **ESM-2 Deep Learning Embeddings**: Passes the sequence through a lightweight HuggingFace transformer model (`facebook/esm2_t6_8M_UR50D`) to capture contextual sequence representations.
        4. **Scoring & Multi-Label GO Prediction**: Evaluates target druggability confidence and predicts Gene Ontology functional classes.
        5. **Groq AI Report Synthesis**: Generates an executive research report with actionable drug discovery recommendations.
        """
    )

st.markdown("---")

# Retrieve Groq API key from Streamlit Secrets
groq_api_key = None
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    pass

if not groq_api_key:
    st.sidebar.warning("⚠️ `GROQ_API_KEY` not found in Streamlit Secrets. AI-generated reports and GO term predictions will be disabled.")

# Input area
seq_input = st.text_area(
    "Protein Sequence Input (FASTA or Raw Amino Acids)",
    value=">Target_Protein_1\nMKTAYIAKQRQISFVKSHFSRQLEERGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALP",
    height=150,
    help="Provide your protein sequence in FASTA format or raw amino acid string."
)

if st.button("Run Pipeline"):
    if not seq_input.strip():
        st.error("Please enter a valid protein sequence.")
    else:
        with st.spinner("Extracting ESM-2 embeddings and computing sequence features..."):
            try:
                metrics = analyze_protein_sequence(seq_input)
                st.success("Sequence analysis complete!")
                
                # 3. Enhanced Metrics with Question Mark Tooltips
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Sequence Length", 
                    f"{metrics['length']} aa", 
                    help="Total number of amino acid residues. Optimal drug targets typically range between 150-800 amino acids."
                )
                col2.metric(
                    "Molecular Weight", 
                    f"{metrics['mol_weight']} Da", 
                    help="Total molecular mass of the protein in Daltons. Smaller proteins (<50,000 Da) are generally easier to handle in expression and manufacturing."
                )
                col3.metric(
                    "GRAVY Index", 
                    f"{metrics['gravy']}", 
                    help="Grand Average of Hydropathy. Negative values indicate hydrophilic proteins often found on surface-accessible regions; positive values indicate hydrophobic proteins."
                )
                col4.metric(
                    "Instability Index", 
                    f"{metrics['instability_index']}", 
                    help="Estimates in vitro stability. Values below 40 are predicted as stable, whereas values above 40 suggest potential instability."
                )
                
                # Tabs for different output categories
                tab1, tab2, tab3 = st.tabs(["🎯 Druggability & Metrics", "🧬 Functional GO Prediction", "📑 AI Research Report"])
                
                with tab1:
                    st.subheader("Druggability Assessment & Classification")
                    score = metrics['druggability_score']
                    st.progress(score)
                    
                    # 3. Explicit Tier Classification
                    if score >= 0.70:
                        tier = "🟢 HIGH DRUGGABILITY"
                        desc = "Promising target with favorable size, hydrophilic surface exposure, and structural stability characteristics suitable for small molecule or biologic binding."
                    elif score >= 0.50:
                        tier = "🟡 MODERATE DRUGGABILITY"
                        desc = "Moderate target viability. May require structural refinement, domain truncation, or targeted mutagenesis to uncover tractable binding pockets."
                    else:
                        tier = "🔴 LOW DRUGGABILITY"
                        desc = "Challenging target profile. Exhibits unfavorable physicochemical characteristics or instability that may hinder successful drug binding pocket formation."

                    st.markdown(f"### Status: {tier}")
                    st.write(f"**Confidence Score:** `{score * 100:.1f}%` — {desc}")
                        
                    st.markdown("---")
                    st.subheader("Amino Acid Composition (%)")
                    df_aa = pd.DataFrame(list(metrics['aa_composition'].items()), columns=["Amino Acid", "Percentage (%)"])
                    st.bar_chart(df_aa.set_index("Amino Acid"))
                
                with tab2:
                    st.subheader("Gene Ontology (GO) Functional Classification")
                    st.markdown("Multi-label prediction of molecular functions, biological processes, and cellular components derived from sequence embeddings.")
                    
                    if not groq_api_key:
                        st.warning("⚠️ Groq API key is missing from Streamlit secrets. Configure `GROQ_API_KEY` in `.streamlit/secrets.toml` to view functional predictions.")
                    else:
                        with st.spinner("Predicting GO terms using model features..."):
                            go_prediction = predict_gene_ontology_terms(metrics, groq_api_key)
                            st.markdown(go_prediction)
                
                with tab3:
                    st.subheader("Executive Research Report")
                    if not groq_api_key:
                        st.warning("⚠️ Groq API key is missing from Streamlit secrets. Configure `GROQ_API_KEY` in `.streamlit/secrets.toml` to view AI reports.")
                    else:
                        with st.spinner("Generating expert report with Groq..."):
                            report = generate_groq_report(metrics, groq_api_key)
                            st.markdown(report)
                            
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
