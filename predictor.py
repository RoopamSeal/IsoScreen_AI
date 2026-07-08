import os
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import config

class ProteinPredictor:
    def __init__(self):
        """Initializes tokenizers and models with automatic CPU/GPU resource handling."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = AutoModel.from_pretrained(config.MODEL_NAME).to(self.device)
        self.model.eval()

    def get_embedding(self, sequence: str) -> np.ndarray:
        """Tokenizes sequences and applies token-wise mean pooling over spatial dimensions."""
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pull output representation maps: Shape [Batch, Sequence_Len, Hidden_Dim]
        last_hidden_state = outputs.last_hidden_state[0]
        
        # Exclude special boundary tokens (<cls> at index 0, <eos> at terminal position)
        if last_hidden_state.shape[0] > 2:
            seq_embeddings = last_hidden_state[1:-1]
        else:
            seq_embeddings = last_hidden_state
            
        # Compute spatial mean pooling across remaining residue sequence dimensions
        mean_embedding = torch.mean(seq_embeddings, dim=0)
        return mean_embedding.cpu().numpy()

    def predict(self, embedding: np.ndarray) -> float:
        """Scores spatial array maps using the serialized downstream Classifier configuration."""
        if not os.path.exists(config.CLASSIFIER_PATH):
            self._train_fallback_classifier()

        classifier = joblib.load(config.CLASSIFIER_PATH)
        reshaped_input = embedding.reshape(1, -1)
        
        # Extract targeted probabilities mapped to the positive classification label index
        probability = classifier.predict_proba(reshaped_input)[0][1]
        return float(probability)

    def _train_fallback_classifier(self):
        """Generates mock patterns to bootstrap structural classifier files automatically."""
        print("[System Info] Serialized weights missing. Instantiating baseline fallback matrix...")
        np.random.seed(42)
        
        # Construct synthetic balancing data to secure initialization matrix conditions
        X_train = np.random.randn(100, config.EMBEDDING_DIM)
        y_train = np.random.randint(0, 2, size=100)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, config.CLASSIFIER_PATH)
