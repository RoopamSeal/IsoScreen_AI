import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

st.set_page_config(page_title="ADMET Property Predictor", page_icon="💊", layout="wide")

st.title("💊 ADMET Property Predictor")
st.markdown("Enter a chemical structure in **SMILES** format to calculate physicochemical properties and drug-likeness.")

# Example SMILES dropdown or text input
example_smiles = st.selectbox(
    "Choose an example molecule or type your own below:",
    ["Select...", "Aspirin (CC(=O)Oc1ccccc1C(=O)O)", "Caffeine (CN1C=NC2=C1C(=O)N(C(=O)N2C)C)", "Paracetamol (CC(=O)Nc1ccc(O)cc1)"]
)

default_val = ""
if "Aspirin" in example_smiles:
    default_val = "CC(=O)Oc1ccccc1C(=O)O"
elif "Caffeine" in example_smiles:
    default_val = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
elif "Paracetamol" in example_smiles:
    default_val = "CC(=O)Nc1ccc(O)cc1"

smiles_input = st.text_input("SMILES String", value=default_val)

if st.button("Calculate ADMET Properties"):
    if not smiles_input.strip():
        st.error("Please enter a valid SMILES string.")
    else:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES string. RDKit could not parse the molecule.")
        else:
            st.success("Molecule successfully parsed!")
            
            # Compute RDKit Descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Molecular Weight", f"{mw:.2f} g/mol", help="Ideal: < 500 Da")
            col2.metric("LogP (Lipophilicity)", f"{logp:.2f}", help="Ideal: < 5.0")
            col3.metric("TPSA", f"{tpsa:.2f} Å²", help="Polar Surface Area (< 140 Å²)")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("H-Bond Donors", h_donors, help="Ideal: ≤ 5")
            col5.metric("H-Bond Acceptors", h_acceptors, help="Ideal: ≤ 10")
            col6.metric("Rotatable Bonds", rot_bonds, help="Ideal: ≤ 10")
            
            st.markdown("---")
            st.subheader("📋 Lipinski's Rule of 5 Evaluation")
            
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if h_donors > 5: violations += 1
            if h_acceptors > 10: violations += 1
            
            if violations == 0:
                st.success("✅ **Drug-Likeness Passed:** 0 violations of Lipinski's Rule of 5. Excellent oral bioavailability profile.")
            elif violations == 1:
                st.warning("⚠️ **Moderate Profile:** 1 violation of Lipinski's Rule of 5. May still be viable with optimization.")
            else:
                st.error(f"❌ **Poor Drug-Likeness:** {violations} violations of Lipinski's Rule of 5. High risk of poor oral absorption.")
