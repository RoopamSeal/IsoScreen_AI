import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="IsoScreen AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility: RDKit Molecule Renderer ---
def render_2d_structure(smiles, width=300, height=300):
    """Parses a SMILES string and returns a PIL Image of the 2D structure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Generate 2D image
            img = Draw.MolToImage(mol, size=(width, height), fitImage=True)
            return img
        else:
            return None
    except Exception as e:
        return None

# --- Mock Data Generation (Simulating Airflow/Backend Output) ---
@st.cache_data
def generate_mock_data(base_smiles, num_isomers=10):
    """
    Generates mock docking and comprehensive ADMET data.
    In production, this reads from Airflow DAG outputs (AutoDock Vina + GNNs).
    """
    np.random.seed(42)
    
    data = []
    # Original Drug Baseline
    data.append({
        "Isomer_ID": "Original Ligand",
        "Type": "Baseline",
        "Binding_Affinity": -7.2, # kcal/mol
        "Metabolic_HalfLife_min": 45,
        "Toxicity_Prob": 0.35,
        "LogP": 3.2,
        "Solubility_LogS": -4.5,
        "CYP450_Inhibition": "High Risk",
        "Lipinski_Rules": "Pass",
        "Developability_Score": 60.5,
        "SMILES": base_smiles # using base for rendering
    })
    
    # Generate Variants
    for i in range(1, num_isomers + 1):
        affinity = round(np.random.uniform(-9.5, -5.5), 2)
        half_life = int(np.random.uniform(20, 150))
        tox = round(np.random.uniform(0.05, 0.8), 2)
        logp = round(np.random.uniform(1.5, 5.5), 2)
        logs = round(np.random.uniform(-6.0, -2.0), 2)
        
        cyp_status = np.random.choice(["Low Risk", "Moderate", "High Risk"], p=[0.5, 0.3, 0.2])
        lipinski = np.random.choice(["Pass", "1 Violation", "2 Violations"], p=[0.7, 0.2, 0.1])
        
        # Calculate mock developability score (higher is better)
        score = round(((-affinity / 10) * 35) + ((half_life / 150) * 30) + ((1 - tox) * 20) + ((5 - logp)/5 * 15), 1)
        
        is_winner = (i == 4) # Hardcode one winner for demonstration
        if is_winner:
            affinity = -9.2
            half_life = 135
            tox = 0.08
            logp = 2.8
            logs = -3.2
            cyp_status = "Low Risk"
            lipinski = "Pass"
            score = 94.5
            
        # For dummy visualization, we append an isotope tag to the SMILES
        dummy_smiles = base_smiles.replace("C", "[13C]", 1) if i % 2 == 0 else base_smiles
            
        data.append({
            "Isomer_ID": f"Iso-D{i}" if i % 2 == 0 else f"Iso-S{i}",
            "Type": "Deuterated / Isotopic" if i % 2 == 0 else "Stereoisomer",
            "Binding_Affinity": affinity,
            "Metabolic_HalfLife_min": half_life,
            "Toxicity_Prob": tox,
            "LogP": logp,
            "Solubility_LogS": logs,
            "CYP450_Inhibition": cyp_status,
            "Lipinski_Rules": lipinski,
            "Developability_Score": score,
            "SMILES": dummy_smiles
        })
        
    return pd.DataFrame(data).sort_values(by="Developability_Score", ascending=False)

# --- Sidebar Inputs ---
st.sidebar.title("🧬 IsoScreen AI")
st.sidebar.markdown("### 1. Ligand Input")
smiles_query = st.sidebar.text_input("Ligand SMILES String", "CC(C)C1=CC=C(C=C1)C(C)C(=O)O")

st.sidebar.markdown("### 2. Target Protein Input")
pdb_file = st.sidebar.file_uploader("Upload Protein (.pdb)", type=['pdb'])
if pdb_file:
    st.sidebar.success(f"Loaded: {pdb_file.name}")
else:
    st.sidebar.info("Awaiting PDB file upload...")

st.sidebar.markdown("---")
st.sidebar.markdown("**Simulation Parameters**")
st.sidebar.caption("• Physics: AutoDock Vina\n• Cheminformatics: RDKit\n• ADMET: GNNs (MoleculeNet)")

# Disable button if PDB is not uploaded to simulate real strict workflow
run_pipeline = st.sidebar.button("Run Global Screening Pipeline", type="primary", disabled=not pdb_file)

if not pdb_file:
    st.sidebar.warning("Please upload a target PDB file to enable screening.")

# --- Main Dashboard ---
st.title("Executive Intelligence Dashboard")
st.markdown("Automated Pre-Clinical Virtual Screening & ADMET Profiling")

if run_pipeline and pdb_file:
    with st.spinner("Executing pipeline: RDKit Expansion ➔ Vina Docking ➔ DeepChem ADMET..."):
        time.sleep(2.5) # Simulate processing time
        df = generate_mock_data(smiles_query)
        st.success("Pipeline execution complete! 11 Variants Screened.")
        
        winner = df.iloc[0]
        baseline = df[df['Isomer_ID'] == 'Original Ligand'].iloc[0]
        
        # --- TOP SECTION: 2D STRUCTURES & PRIMARY METRICS ---
        st.subheader("Lead Optimization Summary")
        col_img1, col_img2, col_mets = st.columns([1, 1, 2])
        
        with col_img1:
            st.markdown("**Original Ligand**")
            img_baseline = render_2d_structure(baseline['SMILES'])
            if img_baseline:
                st.image(img_baseline, use_container_width=True)
            else:
                st.error("Invalid SMILES for rendering.")
                
        with col_img2:
            st.markdown(f"**Top Candidate ({winner['Isomer_ID']})**")
            img_winner = render_2d_structure(winner['SMILES'])
            if img_winner:
                st.image(img_winner, use_container_width=True)
            else:
                st.error("Invalid SMILES for rendering.")

        with col_mets:
            st.markdown("**Core Impact Metrics**")
            m1, m2 = st.columns(2)
            m1.metric("Developability Score", f"{winner['Developability_Score']}/100", 
                      f"+{round(winner['Developability_Score'] - baseline['Developability_Score'], 1)} vs Original")
            m2.metric("Binding Affinity (ΔG)", f"{winner['Binding_Affinity']} kcal/mol", 
                      f"{round(baseline['Binding_Affinity'] - winner['Binding_Affinity'], 2)} kcal/mol tighter", delta_color="inverse")
            
            m3, m4 = st.columns(2)
            m3.metric("Metabolic Half-Life", f"{winner['Metabolic_HalfLife_min']} mins", 
                      f"+{winner['Metabolic_HalfLife_min'] - baseline['Metabolic_HalfLife_min']} mins extension")
            m4.metric("Toxicity Risk", f"{winner['Toxicity_Prob']*100:.1f}%", 
                      f"{round((winner['Toxicity_Prob'] - baseline['Toxicity_Prob'])*100, 1)}% reduction", delta_color="inverse")

        st.markdown("---")
        
        # --- MIDDLE SECTION: ADMET PROFILING & SCATTER ---
        col_scatter, col_radar = st.columns([3, 2])
        
        with col_scatter:
            st.subheader("Affinity vs. Stability Landscape")
            fig_scatter = px.scatter(
                df, 
                x="Binding_Affinity", 
                y="Metabolic_HalfLife_min", 
                color="Type",
                size="Developability_Score",
                hover_name="Isomer_ID",
                hover_data=["LogP", "CYP450_Inhibition"],
                labels={
                    "Binding_Affinity": "Binding Affinity (kcal/mol) [Lower = Better]",
                    "Metabolic_HalfLife_min": "Half-Life (mins) [Higher = Better]"
                },
                color_discrete_map={"Baseline": "#F59E0B", "Deuterated / Isotopic": "#4F46E5", "Stereoisomer": "#06B6D4"}
            )
            # Invert X axis (more negative affinity is better, placed on the right)
            fig_scatter.update_xaxes(autorange="reversed")
            fig_scatter.add_vline(x=baseline['Binding_Affinity'], line_dash="dash", line_color="gray", opacity=0.5)
            fig_scatter.add_hline(y=baseline['Metabolic_HalfLife_min'], line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_radar:
            st.subheader("Multiparameter ADMET Profile")
            st.caption("Comparing Top Candidate vs Baseline (Normalized 0-100 scale, larger area is better)")
            
            # Normalize ADMET metrics for radar chart display (Higher = Better)
            categories = ['Safety (1-Tox)', 'Stability (T1/2)', 'Lipophilicity (Optimal LogP)', 'Solubility (LogS)', 'Binding (-ΔG)']
            
            # Normalize logic for dummy display
            w_tox = (1 - winner['Toxicity_Prob']) * 100
            w_hl = min((winner['Metabolic_HalfLife_min'] / 150) * 100, 100)
            w_logp = 100 - (abs(winner['LogP'] - 2.5) * 20) # Ideal LogP ~ 2.5
            w_sol = min((abs(winner['Solubility_LogS']) / 6) * 100, 100) # Closer to 0 is better, but this is a rough proxy
            w_aff = min((abs(winner['Binding_Affinity']) / 10) * 100, 100)
            
            b_tox = (1 - baseline['Toxicity_Prob']) * 100
            b_hl = min((baseline['Metabolic_HalfLife_min'] / 150) * 100, 100)
            b_logp = 100 - (abs(baseline['LogP'] - 2.5) * 20)
            b_sol = min((abs(baseline['Solubility_LogS']) / 6) * 100, 100)
            b_aff = min((abs(baseline['Binding_Affinity']) / 10) * 100, 100)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                  r=[w_tox, w_hl, w_logp, w_sol, w_aff],
                  theta=categories, fill='toself', name=winner['Isomer_ID'], line_color='#4F46E5'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                  r=[b_tox, b_hl, b_logp, b_sol, b_aff],
                  theta=categories, fill='toself', name='Baseline', line_color='#F59E0B'
            ))
            fig_radar.update_layout(
              polar=dict(radialaxis=dict(visible=False, range=[0, 100])),
              showlegend=True, margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- BOTTOM SECTION: DETAILED ADMET DATAFRAME ---
        st.markdown("---")
        st.subheader("Comprehensive Pipeline Results")
        
        # Select and reorder columns for executive view
        display_df = df[['Isomer_ID', 'Type', 'Binding_Affinity', 'Metabolic_HalfLife_min', 
                         'CYP450_Inhibition', 'LogP', 'Solubility_LogS', 'Lipinski_Rules', 'Toxicity_Prob', 'Developability_Score']]
        
        # Apply conditional formatting
        styled_df = display_df.style\
            .background_gradient(cmap="Blues", subset=["Developability_Score"])\
            .background_gradient(cmap="RdYlGn_r", subset=["Toxicity_Prob"])\
            .format({
                "Toxicity_Prob": "{:.1%}",
                "Binding_Affinity": "{:.2f}",
                "LogP": "{:.2f}",
                "Solubility_LogS": "{:.2f}"
            })
            
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

else:
    # Landing page state
    st.info("👈 **Awaiting Inputs:** Please upload a Target PDB file and provide a Ligand SMILES string in the sidebar, then click **Run Global Screening Pipeline**.")
    
    st.markdown("### How the Pipeline Works")
    st.markdown("""
    1. **Data Ingestion:** PDB files and SMILES strings are verified.
    2. **Generative Chemistry (RDKit):** Computes all stereoisomers and performs targeted isotopic swapping.
    3. **Physics Screening (AutoDock Vina):** Conducts high-throughput molecular docking against the target protein.
    4. **ADMET Prediction (GNNs):** DeepChem neural networks analyze variants for Toxicity, CYP450 metabolism, and Lipinski compliance.
    5. **Executive Output:** Ranks all generated candidates by their composite Developability Score.
    """)