"""
config.py
---------
Central configuration for GraphDrugPred v1 (Sequence-Only Baseline).

Holds model identifiers, file paths, thresholds, and LLM provider
settings used across predictor.py, agent.py, and app.py.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Path where the trained (or mock-trained) classifier head is persisted.
CLASSIFIER_PATH = MODELS_DIR / "druggability_classifier.joblib"

# --------------------------------------------------------------------------
# ESM-2 protein language model
# --------------------------------------------------------------------------
# Lightweight model for fast prototyping / CPU-only environments.
ESM_MODEL_LIGHT = "facebook/esm2_t6_8M_UR50D"        # 320-dim embeddings
# Larger, more accurate model (requires more RAM/VRAM).
ESM_MODEL_LARGE = "facebook/esm2_t33_650M_UR50D"     # 1280-dim embeddings

# Toggle which ESM-2 checkpoint is used by default. Override via the
# GRAPHDRUGPRED_ESM_MODEL environment variable if desired.
ESM_MODEL_NAME = os.environ.get("GRAPHDRUGPRED_ESM_MODEL", ESM_MODEL_LIGHT)

# Embedding dimensionality per checkpoint (hidden_size of the ESM-2 model).
ESM_EMBEDDING_DIMS = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}

EMBEDDING_DIM = ESM_EMBEDDING_DIMS.get(ESM_MODEL_NAME, 320)

# Maximum sequence length (residues) accepted by the pipeline. Longer
# sequences are truncated to keep inference tractable on CPU.
MAX_SEQUENCE_LENGTH = 1022  # ESM-2 positional embedding limit is 1024 (incl. special tokens)

# --------------------------------------------------------------------------
# Classifier / druggability thresholds
# --------------------------------------------------------------------------
# Probability >= this threshold is labeled "Druggable".
DRUGGABILITY_THRESHOLD = 0.5

# Confidence bands used purely for human-readable reporting.
CONFIDENCE_BANDS = {
    "high": 0.80,
    "medium": 0.60,
}

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Valid amino acid alphabet (standard 20 + common ambiguity codes)
# --------------------------------------------------------------------------
# X = unknown, B = Asx, Z = Glx, J = Leu/Ile, U = Selenocysteine, O = Pyrrolysine
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AMINO_ACIDS = set("XBZJUO")
ALL_ACCEPTED_CHARS = VALID_AMINO_ACIDS | AMBIGUOUS_AMINO_ACIDS

# Fraction of ambiguous/unknown residues above which a sequence is rejected.
MAX_AMBIGUOUS_FRACTION = 0.10

# --------------------------------------------------------------------------
# LLM / Agent settings (LangGraph + LangChain)
# --------------------------------------------------------------------------
# GraphDrugPred defaults to Groq's free-tier API (fast, generous free quota)
# for the report-generation node. If no API key is configured, agent.py
# transparently falls back to a deterministic, template-based report so
# the whole application still runs fully offline / out-of-the-box.
LLM_PROVIDER = os.environ.get("GRAPHDRUGPRED_LLM_PROVIDER", "groq")  # "groq" | "ollama" | "none"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.environ.get("GRAPHDRUGPRED_GROQ_MODEL", "llama-3.1-8b-instant")

OLLAMA_MODEL_NAME = os.environ.get("GRAPHDRUGPRED_OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 600

# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
APP_TITLE = "GraphDrugPred — Sequence-Only Druggability Baseline"
APP_ICON = "🧬"
