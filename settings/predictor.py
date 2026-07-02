"""
predictor.py
------------
Core ML logic for GraphDrugPred v1.

Responsibilities:
    1. Load the ESM-2 protein language model + tokenizer (cached, singleton).
    2. Tokenize a raw amino-acid sequence and extract per-residue embeddings.
    3. Mean-pool residue embeddings into a single fixed-length protein vector.
    4. Load (or, if absent, train a mock) lightweight classifier head that
       maps the embedding -> druggability probability.

Design notes:
    - `torch.no_grad()` is used throughout inference to avoid building the
      autograd graph and to conserve memory.
    - The classifier is intentionally simple (RandomForest) so v1 can run
      fully on CPU with no GPU dependency.
    - If no trained classifier weights exist on disk, `load_or_train_classifier`
      synthesizes a small, structured mock dataset in embedding space and
      trains a RandomForest on it. This keeps the app fully functional
      out-of-the-box while making it obvious (via logging + report caveats)
      that the shipped model is a placeholder pending real training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

import config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predictor")


# ==========================================================================
# Data containers
# ==========================================================================
@dataclass
class EmbeddingResult:
    vector: np.ndarray          # shape: (embedding_dim,)
    sequence_length: int
    model_name: str


@dataclass
class PredictionResult:
    prediction_label: str       # "Druggable" | "Non-Druggable"
    probability: float          # P(druggable), in [0, 1]
    confidence: float           # distance from decision boundary, in [0, 1]
    is_mock_model: bool         # True if classifier was trained on synthetic data


# ==========================================================================
# ESM-2 model loading (singleton / cached)
# ==========================================================================
@lru_cache(maxsize=1)
def _load_esm_model(model_name: str = config.ESM_MODEL_NAME):
    """
    Loads and caches the ESM-2 tokenizer + model from Hugging Face.

    Cached with lru_cache so repeated calls (e.g. across Streamlit reruns
    or multiple LangGraph node invocations) do not re-download or
    re-initialize the model.
    """
    from transformers import AutoTokenizer, AutoModel

    logger.info("Loading ESM-2 model '%s' (first call may download weights)...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # inference mode: disables dropout etc.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("ESM-2 model loaded on device: %s", device)

    return tokenizer, model, device


def get_model_bundle(model_name: str = config.ESM_MODEL_NAME):
    """Public accessor for the cached (tokenizer, model, device) tuple."""
    return _load_esm_model(model_name)


# ==========================================================================
# Embedding extraction
# ==========================================================================
def extract_embedding(
    sequence: str,
    model_name: str = config.ESM_MODEL_NAME,
) -> EmbeddingResult:
    """
    Runs a cleaned amino-acid sequence through ESM-2 and mean-pools the
    final hidden layer's per-residue representations into a single
    fixed-length vector.

    Args:
        sequence: Cleaned, uppercase amino-acid sequence (no header, no whitespace).
        model_name: Hugging Face checkpoint identifier to use.

    Returns:
        EmbeddingResult containing the pooled embedding vector.

    Raises:
        ValueError: if the sequence is empty after cleaning.
        RuntimeError: if the underlying model fails to run inference.
    """
    if not sequence:
        raise ValueError("Cannot extract embeddings from an empty sequence.")

    tokenizer, model, device = get_model_bundle(model_name)

    # Truncate overly long sequences to respect ESM-2's positional embedding limit.
    truncated_sequence = sequence[: config.MAX_SEQUENCE_LENGTH]
    if len(truncated_sequence) < len(sequence):
        logger.warning(
            "Sequence length %d exceeds MAX_SEQUENCE_LENGTH=%d; truncating.",
            len(sequence), config.MAX_SEQUENCE_LENGTH,
        )

    try:
        inputs = tokenizer(
            truncated_sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=config.MAX_SEQUENCE_LENGTH + 2,  # + BOS/EOS special tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():  # conserve memory: no autograd graph during inference
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

            # Build a mask that excludes special tokens (BOS/EOS/PAD) from pooling.
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            special_tokens_mask = torch.tensor(
                tokenizer.get_special_tokens_mask(
                    inputs["input_ids"][0].tolist(), already_has_special_tokens=True
                ),
                device=device,
            ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

            residue_mask = attention_mask * (1 - special_tokens_mask)
            residue_mask = residue_mask.float()

            # Mean-pool only over real residue positions.
            summed = (last_hidden_state * residue_mask).sum(dim=1)
            counts = residue_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts  # (1, hidden_dim)

            pooled_vector = mean_pooled.squeeze(0).cpu().numpy()

    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as RuntimeError
        raise RuntimeError(f"ESM-2 inference failed: {exc}") from exc

    return EmbeddingResult(
        vector=pooled_vector,
        sequence_length=len(truncated_sequence),
        model_name=model_name,
    )


# ==========================================================================
# Classifier head: load pre-trained weights, or train a mock fallback
# ==========================================================================
def _synthesize_training_data(n_samples: int = 400, embedding_dim: int = config.EMBEDDING_DIM):
    """
    Generates a small synthetic dataset in embedding space, used ONLY as a
    fallback so the application is functional out-of-the-box before a real
    classifier has been trained on curated druggability labels (e.g. from
    DrugBank / ChEMBL druggable-genome annotations).

    The synthetic labeling rule uses a fixed random linear projection of the
    embedding plus noise, which produces a learnable-but-imperfect signal —
    enough to demonstrate the full pipeline end-to-end.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, embedding_dim))

    # Fixed pseudo-random linear "oracle" direction to generate correlated labels.
    oracle_weights = rng.normal(loc=0.0, scale=1.0, size=(embedding_dim,))
    scores = X @ oracle_weights
    noise = rng.normal(loc=0.0, scale=np.std(scores) * 0.5, size=n_samples)
    noisy_scores = scores + noise

    threshold = np.median(noisy_scores)
    y = (noisy_scores >= threshold).astype(int)

    return X, y


def load_or_train_classifier(
    path=config.CLASSIFIER_PATH,
    embedding_dim: int = config.EMBEDDING_DIM,
) -> tuple:
    """
    Loads a persisted classifier from disk if available; otherwise trains a
    RandomForest on synthetic data and persists it, so the app can run
    immediately without requiring the user to supply pre-trained weights.

    Returns:
        (classifier, is_mock_model: bool)
    """
    if path.exists():
        try:
            bundle = joblib.load(path)
            classifier = bundle["classifier"]
            trained_dim = bundle.get("embedding_dim", embedding_dim)
            is_mock = bundle.get("is_mock_model", False)

            if trained_dim != embedding_dim:
                logger.warning(
                    "Persisted classifier expects embedding_dim=%d but current "
                    "ESM_MODEL_NAME produces embedding_dim=%d. Retraining mock "
                    "classifier to match current configuration.",
                    trained_dim, embedding_dim,
                )
            else:
                logger.info("Loaded persisted classifier from %s (mock=%s).", path, is_mock)
                return classifier, is_mock
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load classifier at %s (%s). Retraining mock model.", path, exc)

    logger.info(
        "No compatible classifier weights found at %s. Training a fallback "
        "mock RandomForest classifier on synthetic embedding-space data so "
        "the pipeline is immediately runnable. Replace this with a model "
        "trained on real druggability labels for production use.",
        path,
    )

    X, y = _synthesize_training_data(embedding_dim=embedding_dim)
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    classifier.fit(X, y)

    bundle = {
        "classifier": classifier,
        "embedding_dim": embedding_dim,
        "is_mock_model": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Mock classifier trained and saved to %s.", path)

    return classifier, True


# ==========================================================================
# End-to-end prediction
# ==========================================================================
def predict_druggability(
    embedding: np.ndarray,
    classifier: Optional[object] = None,
    is_mock_model: Optional[bool] = None,
) -> PredictionResult:
    """
    Runs the classification head on a pooled embedding vector and derives
    a human-interpretable prediction, probability, and confidence score.

    If `classifier` is not supplied, it is loaded (or mock-trained) lazily.
    """
    if classifier is None:
        classifier, is_mock_model = load_or_train_classifier(embedding_dim=embedding.shape[-1])

    embedding_2d = embedding.reshape(1, -1)
    probabilities = classifier.predict_proba(embedding_2d)[0]  # [P(class0), P(class1)]
    prob_druggable = float(probabilities[1])

    label = "Druggable" if prob_druggable >= config.DRUGGABILITY_THRESHOLD else "Non-Druggable"

    # Confidence: how far the probability sits from the decision boundary (0.5),
    # rescaled to [0, 1] where 1.0 = maximally confident.
    confidence = float(abs(prob_druggable - 0.5) * 2)

    return PredictionResult(
        prediction_label=label,
        probability=prob_druggable,
        confidence=confidence,
        is_mock_model=bool(is_mock_model),
    )
