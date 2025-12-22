import numpy as np
from faiss_utils import init_faiss

MATCH_THRESHOLD = 0.50


class ArcFaceRecognizer:
    def __init__(self):
        """Initialize the recognizer with FAISS index and metadata."""
        self.index, self.metadata = init_faiss()
    
    def recognize(self, embedding):
        """
        Recognize a face embedding using FAISS search.
        
        Args:
            embedding: Face embedding vector (numpy array)
            
        Returns:
            dict with keys:
                - status: "MATCH" or "NO_MATCH"
                - name: Matched person's name (if status is "MATCH")
                - confidence: Similarity score
        """
        # Reshape embedding for FAISS search
        vec = np.asarray(embedding, dtype="float32").reshape(1, -1)
        
        # Search for nearest neighbor
        D, I = self.index.search(vec, k=1)
        
        # Check if valid match found
        if I[0][0] == -1 or D[0][0] < MATCH_THRESHOLD:
            return {
                "status": "NO_MATCH",
                "name": None,
                "confidence": D[0][0] if I[0][0] != -1 else 0.0
            }
        
        # Return matched identity
        return {
            "status": "MATCH",
            "name": self.metadata[I[0][0]],
            "confidence": D[0][0]
        }
