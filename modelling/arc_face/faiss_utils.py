import os
import pickle
import faiss
import numpy as np

# Always store DB relative to this file's directory to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "vector_db")
INDEX_PATH = os.path.join(DB_DIR, "arcface.index")
META_PATH = os.path.join(DB_DIR, "metadata.pkl")
EMBED_DIM = 512


def init_faiss():
    os.makedirs(DB_DIR, exist_ok=True)

    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            if os.path.exists(META_PATH):
                with open(META_PATH, "rb") as f:
                    metadata = pickle.load(f)
            else:
                metadata = {}
        except (RuntimeError, EOFError):
            # Index file is corrupted, create a new one but keep metadata if available
            print("Warning: Index file corrupted. Creating a new index.")
            base = faiss.IndexFlatIP(EMBED_DIM)
            index = faiss.IndexIDMap(base)
            if os.path.exists(META_PATH):
                try:
                    with open(META_PATH, "rb") as f:
                        metadata = pickle.load(f)
                except Exception:
                    metadata = {}
            else:
                metadata = {}
    else:
        base = faiss.IndexFlatIP(EMBED_DIM)
        index = faiss.IndexIDMap(base)
        metadata = {}

    return index, metadata


def save_faiss(index, metadata):
    # Write index and metadata atomically-ish
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def add_embedding(index, metadata, embedding_id, embedding, name):
    # Validate dimension
    emb = np.asarray(embedding, dtype="float32")
    if emb.ndim == 2:
        emb = emb.reshape(-1)
    if emb.shape[-1] != EMBED_DIM:
        raise ValueError(f"Embedding dim {emb.shape[-1]} mismatch; expected {EMBED_DIM}")

    # Normalize for inner-product similarity (optional but recommended for ArcFace)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    vec = emb.reshape(1, -1)
    index.add_with_ids(vec, np.array([embedding_id], dtype="int64"))
    metadata[embedding_id] = name


def reset_faiss():
    """Delete all vector db files to start fresh."""
    for p in (INDEX_PATH, META_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"[INFO] Removed {p}")
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")
