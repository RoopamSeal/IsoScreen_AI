import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio.SeqUtils import ProtParam
from groq import Groq
from config import ESM_MODEL_NAME, GROQ_MODEL_NAME

def load_esm_model():
    """Loads the lightweight ESM-2 model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME)
    model.eval()
    return tokenizer, model

def clean_fasta_sequence(raw_input: str) -> str:
    """
    Cleans raw input or FASTA formatted sequence. 
    Removes header lines (starting with '>') and strips all whitespace/newlines.
    """
    lines = raw_input.strip().splitlines()
    sequence_lines = [line.strip() for line in lines if line.strip() and not line.startswith(">")]
    clean_seq = "".join(sequence_lines).upper()
    return clean_seq

def analyze_protein_sequence(sequence: str):
    """
    Performs BioPython physicochemical analysis, amino acid composition,
    and extracts ESM-2 embedding features to evaluate druggability and function.
    """
    clean_seq = clean_fasta_sequence(sequence)
    
    if not clean_seq:
        raise ValueError("The provided sequence is empty or invalid after cleaning.")

    # 1. BioPython Analysis & Amino Acid Composition (AAC)
    analysed_seq = ProtParam.ProteinAnalysis(clean_seq)
    length = len(clean_seq)
    mol_weight = analysed_seq.molecular_weight()
    instability_index = analysed_seq.instability_index()
    gravy = analysed_seq.gravy()
    aromaticity = analysed_seq.aromaticity()
    
    # Calculate amino acid percentages
    aa_counts = analysed_seq.count_amino_acids()
    aa_composition = {aa: round((count / length) * 100, 2) for aa, count in aa_counts.items()}
    
    # 2. ESM-2 Embedding Feature Extraction
    tokenizer, model = load_esm_model()
    inputs = tokenizer(clean_seq, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pool sequence representations across hidden dimensions
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding_norm = float(np.linalg.norm(embeddings))
    
    # 3. Druggability Heuristic Scoring Model
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
        "druggability_score": round(score, 2),
        "aa_composition": aa_composition
    }
    
    return metrics

def predict_gene_ontology_terms(metrics: dict, api_key: str) -> str:
    """Uses Groq and sequence metrics/embeddings to predict Gene Ontology (GO) terms."""
    if not api_key:
        return "⚠️ Groq API key is required to generate GO term predictions."
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert bioinformatics functional annotator. Based on the following protein sequence features and ESM-2 deep learning representations:
    - Length: {metrics['length']} aa
    - Molecular Weight: {metrics['mol_weight']} Da
    - Gravy: {metrics['gravy']}
    - Aromaticity: {metrics['aromaticity']}
    - Embedding Norm: {metrics['embedding_norm']}
    - Amino Acid Composition Summary: High-level physicochemical property profile.

    Perform a multi-label functional classification prediction. Provide:
    1. Predicted Molecular Function (MF) GO terms with confidence levels (e.g., ATP binding, catalytic activity, protein binding).
    2. Predicted Biological Process (BP) GO terms (e.g., metabolic process, signal transduction, regulation of transcription).
    3. Predicted Cellular Component (CC) GO terms (e.g., cytoplasm, membrane, nucleus).
    Format the response clearly with bullet points and estimated confidence percentages.
    """
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an advanced bioinformatics function and ontology prediction agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error predicting GO terms: {e}"

def generate_groq_report(metrics: dict, api_key: str) -> str:
    """Sends protein metrics and analysis to Groq API to generate an executive research report."""
    if not api_key:
        return "⚠️ Groq API key is missing."
    
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert bioinformatics and drug discovery scientist. Analyze the following target protein screening metrics:
    - Sequence Length: {metrics['length']} amino acids
    - Molecular Weight: {metrics['mol_weight']} Da
    - GRAVY (Hydropathy): {metrics['gravy']} 
    - Instability Index: {metrics['instability_index']} 
    - Aromaticity: {metrics['aromaticity']}
    - ESM-2 Embedding Norm: {metrics['embedding_norm']}
    - Calculated Druggability Confidence Score: {metrics['druggability_score'] * 100:.1f}%

    Generate a structured, professional drug discovery research report with the following sections:
    1. Executive Summary & Target Viability
    2. Physicochemical & Structural Interpretation
    3. Binding Pocket & Druggability Feasibility
    4. Recommended Experimental Validation Steps
    """
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
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
