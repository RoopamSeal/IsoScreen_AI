import streamlit as st
import pandas as pd
import sys
import os

# Add the parent folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.smiles_standardizer import standardize_smiles
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="Hybrid ADMET Prediction Platform", layout="wide")

st.title("Hybrid ADMET Prediction Platform")
st.write("Standardize and visualize molecules via direct input or CSV upload.")

# --- INPUT SECTION ---
input_method = st.radio("Choose Input Method:", ("Text Input", "CSV Upload"))
smiles_list = []

if input_method == "Text Input":
    text_input = st.text_area("Enter SMILES strings (one per line):", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O")
    if text_input:
        # Split by newlines and remove empty spaces/lines
        smiles_list = [s.strip() for s in text_input.split('\n') if s.strip()]

else:
    uploaded_file = st.file_uploader("Upload CSV with a 'smiles' column", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "smiles" in df.columns:
                smiles_list = df["smiles"].dropna().tolist()
            else:
                st.error("CSV must contain a 'smiles' column")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# --- PROCESSING SECTION ---
if smiles_list:
    standardized_mols = []
    invalid_smiles = []

    for smi in smiles_list:
        mol = standardize_smiles(smi)
        if mol is None:
            invalid_smiles.append(smi)
        standardized_mols.append(mol)

    # --- SUMMARY ---
    st.subheader("Molecule Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Input", len(smiles_list))
    col2.metric("Valid Molecules", len([m for m in standardized_mols if m is not None]))
    col3.metric("Invalid detected", len(invalid_smiles))

    if invalid_smiles:
        st.warning(f"Invalid SMILES encountered: {', '.join(invalid_smiles)}")

    # --- PREVIEW ---
    st.subheader("Molecule Preview")
    
    # Filter out the Nones so we only try to display valid molecules
    # We zip them so the caption always matches the right image
    valid_data = [
        (s, m) for s, m in zip(smiles_list, standardized_mols) if m is not None
    ]
    
    if valid_data:
        cols = st.columns(3) 
        for idx, (smi, mol) in enumerate(valid_data):
            with cols[idx % 3]:
                # RDKit drawing is now safe because 'mol' is guaranteed to exist
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption=smi)
    else:
        st.info("No valid molecules to display.")
