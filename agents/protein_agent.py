import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio.SeqUtils import ProtParam
from groq import Groq

# Load lightweight ESM-2 model and tokenizer globally (or cached)
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def load_esm_model():
    """Loads the lightweight ESM-2 model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def clean_fasta_sequence(sequence: str) -> str:
    """Removes FASTA header lines (starting with '>') and whitespace."""
    lines = sequence.strip().splitlines()
    # Filter out lines that start with '>' and empty lines
    seq_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith(">")]
    return "".join(seq_lines).upper()

def analyze_protein_sequence(sequence: str):
    """
    Performs BioPython physicochemical analysis and extracts 
    ESM-2 embedding features to compute a druggability confidence score.
    """
    clean_seq = clean_fasta_sequence(sequence)
    
    if not clean_seq:
        raise ValueError("The provided sequence is empty or contains only FASTA headers.")
    
    # 1. BioPython Analysis
    analysed_seq = ProtParam.ProteinAnalysis(clean_seq)
    length = len(clean_seq)
    mol_weight = analysed_seq.molecular_weight()
    instability_index = analysed_seq.instability_index()
    gravy = analysed_seq.gravy()
    aromaticity = analysed_seq.aromaticity()
    
    # 2. ESM-2 Embedding Feature Extraction
    tokenizer, model = load_esm_model()
    inputs = tokenizer(clean_seq, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pool sequence representations across hidden dimensions
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding_norm = float(np.linalg.norm(embeddings))
    
    # 3. ML / Heuristic Scoring Model
    score = 0.4
    if 150 <= length <= 800:
        score += 0.2
    if gravy < 0:
        score += 0.15
    if instability_index < 40:
        score += 0.15
    if embedding_norm > 5.0:
        score += 0.1
        
    score = min(score, 0.95) # Cap confidence at 95%
    
    metrics = {
        "length": length,
        "mol_weight": round(mol_weight, 2),
        "instability_index": round(instability_index, 2),
        "gravy": round(gravy, 2),
        "aromaticity": round(aromaticity, 4),
        "embedding_norm": round(embedding_norm, 2),
        "druggability_score": round(score, 2)
    }
    
    return metrics

def generate_groq_report(metrics: dict, api_key: str) -> str:
    """Sends protein metrics and analysis to Groq API to generate an executive research report."""
    if not api_key:
        return "⚠️ Groq API key is missing. Please enter your API key in the sidebar to generate the AI report."
    
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert bioinformatics and drug discovery scientist. Analyze the following target protein screening metrics:
    - Sequence Length: {metrics['length']} amino acids
    - Molecular Weight: {metrics['mol_weight']} Da
    - GRAVY (Hydropathy): {metrics['gravy']} (Negative = Hydrophilic / Surface Accessible)
    - Instability Index: {metrics['instability_index']} (< 40 is stable)
    - Aromaticity: {metrics['aromaticity']}
    - ESM-2 Embedding Norm: {metrics['embedding_norm']}
    - Calculated Druggability Confidence Score: {metrics['druggability_score'] * 100:.1f}%

    Generate a structured, professional drug discovery research report with the following sections:
    1. Executive Summary & Target Viability
    2. Physicochemical & Structural Interpretation
    3. Binding Pocket & Druggability Feasibility
    4. Recommended Experimental Validation Steps (e.g., AlphaFold 3 / molecular docking)
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a specialized AI assistant for biopharmaceutical research."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating Groq report: {e}"
