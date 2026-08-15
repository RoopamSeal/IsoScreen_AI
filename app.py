import streamlit as st
from config import APP_TITLE, APP_ICON

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown("### Accelerating Early-Stage Drug Discovery with AI & Cheminformatics")

st.info(
    "Welcome to **IsoScreenAI**! Use the sidebar to navigate between tools:\n"
    "1. **Protein Druggability Predictor**: Evaluate protein targets using sequence analysis & ESM-2 embeddings.\n"
    "2. **ADMET Property Predictor**: Calculate molecular properties and pharmacokinetic profiles."
)

st.image("https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=1200&q=80", 
         caption="Early Drug Discovery Pipeline", width="stretch")

st.markdown("---")
st.subheader("💡 About the MVP")
st.markdown(
    "This platform is designed for researchers to perform rapid in-silico screening before moving to expensive "
    "wet-lab experiments. Select a tool from the sidebar to begin."
)
