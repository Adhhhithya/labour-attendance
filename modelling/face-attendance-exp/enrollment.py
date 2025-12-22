"""
enrollment.py
Handles loading known face encodings and names from a pickle file.
"""
import pickle

ENCODINGS_FILE = "C:\\Users\\Admin\\OneDrive\\Documents\\PEP intern\\labour-attendance\\Backend\\encodings.pkl"


def load_encodings(path: str = ENCODINGS_FILE):
    """
    Load known face encodings + names from a pickle file.
    
    Args:
        path: Path to the encodings pickle file.
        
    Returns:
        Tuple of (known_encodings, known_names).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    return known_encodings, known_names
