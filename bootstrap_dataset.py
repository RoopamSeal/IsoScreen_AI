import os
import time
import requests
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from predictor import ProteinPredictor
import config

# ==========================================
# 1. CURATED BASELINE TARGETS (UniProt IDs)
# ==========================================
# Positives: Well-established drug targets (GPCRs, Kinases, Ion Channels)
DRUGGABLE_TARGETS = [
    "P00533", # EGFR (Epidermal growth factor receptor - Kinase)
    "P28223", # HTR2A (5-hydroxytryptamine receptor 2A - GPCR)
    "P14416", # DRD2 (D(2) dopamine receptor - GPCR)
    "P15056", # BRAF (Serine/threonine-protein kinase B-raf)
    "Q12809", # KCNH2 (Potassium voltage-gated channel)
    "P08183", # MDR1 (Multidrug resistance protein 1)
    "P02708", # CHRNA1 (Acetylcholine receptor subunit alpha)
]

# Negatives: Traditionally undruggable or purely structural/housekeeping proteins
NON_DRUGGABLE_TARGETS = [
    "P04264", # KRT2 (Keratin, type II cytoskeletal 2 - Structural)
    "P68133", # ACTA1 (Actin, alpha skeletal muscle - Structural)
    "P01106", # MYC (Myc proto-oncogene protein - Transcription Factor)
    "P04637", # P53 (Cellular tumor antigen p53 - Transcription Factor)
    "P62241", # RS8 (40S ribosomal protein S8)
    "P84098", # RL19 (60S ribosomal protein L19)
    "P02452", # CO1A1 (Collagen alpha-1(I) chain)
]

def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetches and parses a raw FASTA sequence from the UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Skip the >Header line and join the amino acid sequence
        lines = response.text.strip().split('\n')
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
        return sequence
    else:
        print(f"[!] Failed to fetch {uniprot_id}. HTTP Status: {response.status_code}")
        return None

def build_dataset():
    print("Initializing ESM-2 Predictor Engine (This may take a moment to load into memory)...")
    predictor = ProteinPredictor()
    
    X = []
    y = []
    
    print("\n--- Processing Druggable Targets (Class 1) ---")
    for uid in DRUGGABLE_TARGETS:
        print(f"Fetching and embedding {uid}...")
        seq = fetch_uniprot_sequence(uid)
        if seq:
            emb = predictor.get_embedding(seq)
            X.append(emb)
            y.append(1) # Positive Label
            time.sleep(0.5) # Be polite to the UniProt API rate limits
            
    print("\n--- Processing Non-Druggable Targets (Class 0) ---")
    for uid in NON_DRUGGABLE_TARGETS:
        print(f"Fetching and embedding {uid}...")
        seq = fetch_uniprot_sequence(uid)
        if seq:
            emb = predictor.get_embedding(seq)
            X.append(emb)
            y.append(0) # Negative Label
            time.sleep(0.5)
            
    # Matrix Transformation
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset compiled successfully. Feature Matrix Shape: {X.shape}")
    print("Training Baseline Random Forest Classifier...")
    
    # Train the classifier using balanced class weights to account for any slight list discrepancies 
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    
    # Overwrite the dummy file with actual biological weights
    joblib.dump(clf, config.CLASSIFIER_PATH)
    print(f"Success! Scientific model weights saved to {config.CLASSIFIER_PATH}")

if __name__ == "__main__":
    build_dataset()
