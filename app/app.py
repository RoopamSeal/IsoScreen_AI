import streamlit as st
import pandas as pd
from utils.smiles_standardizer import standardize_smiles


st.title("Hybrid ADMET Prediction Platform")

uploaded = st.file_uploader("Upload SMILES CSV")


if uploaded:

    df = pd.read_csv(uploaded)

    smiles = df["smiles"].tolist()

    st.write("Loaded molecules:", len(smiles))
