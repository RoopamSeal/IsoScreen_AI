import streamlit as st

st.set_page_config(
    page_title="IsoScreenAI | Early Drug Discovery",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 IsoScreenAI: Bioinformatics Assistant")
st.markdown("### Accelerating Early-Stage Drug Discovery with AI & Cheminformatics")

st.info(
    "Welcome to **IsoScreenAI**! Use the sidebar to navigate between tools:\n"
    "1. **Protein Druggability Predictor**: Evaluate protein targets using sequence analysis.\n"
    "2. **ADMET Property Predictor**: Calculate molecular properties and pharmacokinetic profiles."
)

st.image("https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=1200&q=80", 
         caption="Early Drug Discovery Pipeline", use_container_width=True)

st.markdown("---")
st.subheader("💡 About the MVP")
st.markdown(
    "This platform is designed for researchers to perform rapid in-silico screening before moving to expensive "
    "wet-lab experiments. Select a tool from the sidebar to begin."
)
