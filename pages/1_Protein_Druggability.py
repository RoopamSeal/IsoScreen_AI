import streamlit as st
from Bio.SeqUtils import ProtParam
import pandas as pd

st.set_page_config(page_title="Protein Druggability Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Protein Druggability Predictor")
st.markdown("Paste your target protein sequence in FASTA or raw format to evaluate its druggability potential.")

# Input text area
seq_input = st.text_area(
    "Protein Sequence (Amino Acids)",
    value="MKTAYIAKQRQISFVKSHFSRQLEERGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALP",
    height=150
)

if st.button("Evaluate Druggability"):
    if not seq_input.strip():
        st.error("Please enter a valid protein sequence.")
    else:
        # Clean sequence (remove whitespace/newlines)
        clean_seq = "".join(seq_input.upper().split())
        
        try:
            # Analyze sequence using BioPython
            analysed_seq = ProtParam.ProteinAnalysis(clean_seq)
            
            # Metrics
            length = len(clean_seq)
            mol_weight = analysed_seq.molecular_weight()
            instability_index = analysed_seq.instability_index()
            gravy = analysed_seq.gravy() # Grand average of hydropathy
            aromaticity = analysed_seq.aromaticity()
            
            st.success("Analysis complete!")
            
            # Display Metrics in Columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sequence Length", f"{length} aa")
            col2.metric("Molecular Weight", f"{mol_weight:.2f} Da")
            col3.metric("GRAVY Index", f"{gravy:.2f}", help="Negative = Hydrophilic, Positive = Hydrophobic")
            col4.metric("Instability Index", f"{instability_index:.2f}", help="< 40 is predicted as stable")
            
            st.markdown("---")
            st.subheader("📊 Druggability Assessment")
            
            # Heuristic scoring simulation for MVP
            score = 0.5
            if 150 <= length <= 800:
                score += 0.2  # Optimal drug target size range
            if gravy < 0:
                score += 0.15 # Surface accessible regions often hydrophilic
            if instability_index < 40:
                score += 0.15
                
            score = min(score, 0.95) # Cap at 95%
            
            st.progress(score)
            st.write(f"**Estimated Druggability Confidence Score:** `{score * 100:.1f}%`")
            
            if score > 0.7:
                st.markdown("🟢 **High Promising Target:** Exhibits favorable molecular weight and stability characteristics for binding pocket formation.")
            else:
                st.markdown("🟡 **Moderate/Challenging Target:** May require structural refinement or domain filtering (e.g., transmembrane domains).")
                
        except Exception as e:
            st.error(f"Error processing sequence. Ensure valid single-letter amino acid codes. Details: {e}")
