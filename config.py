import os

# Model Parameters
# Defaulting to the lightweight 8M model for robust out-of-the-box local execution.
# Swap to "facebook/esm2_t33_650M_UR50D" and set EMBEDDING_DIM = 1280 for production accuracy.
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBEDDING_DIM = 320

# Directory and Persistence Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(BASE_DIR, "druggability_classifier.joblib")

# Classification Thresholds
DRUGGABILITY_THRESHOLD = 0.5
