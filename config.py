"""
===========================================================
GraphDrugPred / IsoScreenAI
Configuration File
===========================================================

Central configuration for the entire application.

Author : Roopam Seal
Version: 2.0
"""

import os
import torch
from pathlib import Path

# ===========================================================
# PROJECT DIRECTORIES
# ===========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "hf_cache"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

CLASSIFIER_PATH = MODEL_DIR / "druggability_classifier.joblib"

# Automatically create directories if they do not exist
for directory in [MODEL_DIR, CACHE_DIR, REPORT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===========================================================
# DEVICE CONFIGURATION
# ===========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================
# HUGGINGFACE MODEL CONFIGURATION
# ===========================================================

# Lightweight ESM-2 model (recommended for Streamlit Cloud)
MODEL_NAME = os.getenv(
    "ESM_MODEL",
    "facebook/esm2_t6_8M_UR50D"
)

# Embedding dimension for each supported model
EMBEDDING_DIMENSIONS = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280
}

EMBEDDING_DIM = EMBEDDING_DIMENSIONS[MODEL_NAME]

MAX_SEQUENCE_LENGTH = 1024

# ===========================================================
# CLASSIFICATION
# ===========================================================

DRUGGABILITY_THRESHOLD = 0.50

# ===========================================================
# RANDOM SEED
# ===========================================================

RANDOM_STATE = 42

# ===========================================================
# RANDOM FOREST FALLBACK CLASSIFIER
# ===========================================================

RF_N_ESTIMATORS = 200

# ===========================================================
# LANGGRAPH
# ===========================================================

MAX_RETRIES = 3

# ===========================================================
# GROQ LLM
# ===========================================================

LLM_MODEL = "llama-3.3-70b-versatile"

LLM_TEMPERATURE = 0.1

# ===========================================================
# PLOTTING
# ===========================================================

PAE_MAX_SIZE = 200

# Prevents memory explosion for long proteins.
# Example:
# 500 aa -> 500x500 matrix
# 2000 aa -> limited to 200x200

# ===========================================================
# VALID AMINO ACIDS
# ===========================================================

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)

# ===========================================================
# LOGGING
# ===========================================================

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "isoscreenai.log"

# ===========================================================
# REPORT SETTINGS
# ===========================================================

REPORT_TITLE = "IsoScreenAI Target Assessment Report"

REPORT_AUTHOR = "GraphDrugPred"

# ===========================================================
# APP SETTINGS
# ===========================================================

APP_NAME = "IsoScreenAI"

APP_ICON = "🧬"

PAGE_LAYOUT = "wide"

# ===========================================================
# FEATURE FLAGS
# ===========================================================

ENABLE_REPORT_GENERATION = True

ENABLE_STRUCTURAL_SIMULATION = True

ENABLE_PAE_VISUALIZATION = True

ENABLE_RADAR_CHART = True

ENABLE_GAUGE_CHART = True

# ===========================================================
# VERSION
# ===========================================================

VERSION = "2.0.0"
