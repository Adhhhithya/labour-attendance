import numpy as np
from faiss_utils import init_faiss

MATCH_THRESHOLD = 0.50  # conservative for ArcFace + IP


class ArcFaceRecognizer:
    def __init__(self):
        self.index, self.metadata = init_faiss()

    @staticmethod
    def _normalize(vec):
        return vec / np.linalg.norm(vec)

    def recognize(self, embedding):
        """
        embedding: numpy array (512,)
        returns: dict with recognition result
        """
        vec = np.asarray(embedding, dtype="float32").reshape(1, -1)
        vec = self._normalize(vec)

        D, I = self.index.search(vec, k=1)

        if I[0][0] == -1:
            return {
                "status": "UNKNOWN",
                "name": None,
                "confidence": 0.0,
                "embedding_id": None,
            }

        score = float(D[0][0])
        embedding_id = int(I[0][0])

        if score < MATCH_THRESHOLD:
            return {
                "status": "UNKNOWN",
                "name": None,
                "confidence": score,
                "embedding_id": None,
            }

        return {
            "status": "MATCH",
            "name": self.metadata.get(embedding_id),
            "confidence": score,
            "embedding_id": embedding_id,
        }
