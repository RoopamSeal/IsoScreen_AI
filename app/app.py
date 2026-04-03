import streamlit as st
import pandas as pd
from utils.smiles_standardizer import standardize_smiles
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="Hybrid ADMET Prediction Platform", layout="wide")

st.title("Hybrid ADMET Prediction Platform")
st.write(
    """
Upload a CSV file containing SMILES strings.
The app will standardize the molecules and display a summary.
"""
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV with a 'smiles' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if "smiles" not in df.columns:
        st.error("CSV must contain a 'smiles' column")
        st.stop()

    # Standardize SMILES
    smiles_list = df["smiles"].tolist()
    standardized_mols = []
    invalid_smiles = []

    for smi in smiles_list:
        mol = standardize_smiles(smi)
        if mol is None:
            invalid_smiles.append(smi)
        standardized_mols.append(mol)

    st.subheader("Molecule Summary")
    st.write(f"Total molecules uploaded: {len(smiles_list)}")
    st.write(f"Valid molecules: {len([m for m in standardized_mols if m is not None])}")
    st.write(f"Invalid SMILES detected: {len(invalid_smiles)}")

    if invalid_smiles:
        st.warning(f"Invalid SMILES: {invalid_smiles}")

    # Display molecules
    st.subheader("Molecule Preview")
    for mol in standardized_mols:
        if mol:
            st.image(Draw.MolToImage(mol, size=(300, 300)))

