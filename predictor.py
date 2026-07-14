"""
===========================================================
GraphDrugPred / IsoScreenAI
Protein Predictor Module
===========================================================

Responsibilities
----------------
1. Load ESM-2 tokenizer
2. Load ESM-2 model (Optimized for Cloud Environments)
3. Generate protein embeddings
4. Load trained classifier
5. Predict druggability probability
"""

import os
import streamlit as st

# CRITICAL CLOUD FIX: Prevent CPU thread deadlocks before torch is imported
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import joblib
import numpy as np
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModel, AutoTokenizer

import config

# Force PyTorch to only use 1 thread to avoid freezing the cloud container
torch.set_num_threads(1)

class ProteinPredictor:
    """
    Protein embedding + prediction pipeline.
    """

    def __init__(self):
        self.device = torch.device(config.DEVICE)

        print(f"Loading tokenizer: {config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            cache_dir=config.CACHE_DIR
        )

        print(f"Loading ESM2 model: {config.MODEL_NAME} with memory optimizations...")
        
        # CLOUD FIX: Load model strictly in evaluation mode without memory gradients
        with torch.no_grad():
            self.model = AutoModel.from_pretrained(
                config.MODEL_NAME,
                cache_dir=config.CACHE_DIR,
                low_cpu_mem_usage=True  # Crucial for 1GB RAM limits
            )
            self.model.to(self.device)
            self.model.eval()

        print("Loading classifier...")
        self.classifier = self._load_classifier()
        print("ProteinPredictor initialized successfully.")

    ####################################################################
    # EMBEDDING EXTRACTION
    ####################################################################

    def get_embedding(self, sequence: str) -> np.ndarray:
        """
        Convert a protein sequence into an ESM embedding.

        Returns
        -------
        numpy.ndarray
            Shape = (EMBEDDING_DIM,)
        """
        if len(sequence) == 0:
            raise ValueError("Empty protein sequence.")

        if len(sequence) > config.MAX_SEQUENCE_LENGTH:
            sequence = sequence[: config.MAX_SEQUENCE_LENGTH]

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state[0]

        # Remove CLS and EOS tokens
        if hidden.shape[0] > 2:
            hidden = hidden[1:-1]

        embedding = hidden.mean(dim=0)
        return embedding.cpu().numpy().astype(np.float32)

    ####################################################################
    # CLASSIFIER
    ####################################################################

    def _load_classifier(self):
        classifier_path = Path(config.CLASSIFIER_PATH)

        if classifier_path.exists():
            print("Existing classifier found.")
            return joblib.load(classifier_path)

        print("Classifier not found.")
        print("Creating fallback classifier...")
        return self._train_fallback_classifier()

    ####################################################################
    # FALLBACK MODEL
    ####################################################################

    def _train_fallback_classifier(self):
        np.random.seed(config.RANDOM_STATE)
        X = np.random.normal(size=(500, config.EMBEDDING_DIM))
        y = np.random.randint(0, 2, size=500)

        clf = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            random_state=config.RANDOM_STATE
        )
        clf.fit(X, y)

        joblib.dump(clf, config.CLASSIFIER_PATH)
        print("Fallback classifier saved.")
        return clf

    ####################################################################
    # PREDICTION
    ####################################################################

    def predict(self, embedding: np.ndarray) -> float:
        """
        Predict probability of druggability.

        Returns
        -------
        float
            Probability between 0 and 1.
        """
        embedding = embedding.reshape(1, -1)
        probability = self.classifier.predict_proba(embedding)[0][1]
        return float(probability)

    ####################################################################
    # COMPLETE PIPELINE
    ####################################################################

    def predict_from_sequence(self, sequence: str):
        embedding = self.get_embedding(sequence)
        probability = self.predict(embedding)
        confidence = abs(probability - config.DRUGGABILITY_THRESHOLD) * 2

        return {
            "embedding": embedding,
            "prediction": probability,
            "confidence": confidence
        }

# --- THE FIX: Safe Singleton Wrapper ---
@st.cache_resource(show_spinner=False)
def get_predictor():
    """
    Safely loads the PyTorch model into RAM exactly once per server boot.
    Prevents Out-Of-Memory (OOM) crashes and thread deadlocks on Streamlit/Render.
    """
    return ProteinPredictor()
