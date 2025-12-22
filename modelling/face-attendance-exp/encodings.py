import os
import face_recognition
import pickle
import sys

ENCODINGS_FILE = "encodings.pkl"

def load_existing_encodings():
    """Load existing encodings from file if it exists."""
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                return data.get("encodings", []), data.get("names", [])
        except Exception as e:
            print(f"[WARN] Could not load existing encodings: {e}")
    return [], []

def register_new_face(name, image_path):
    """Register a single face from an image path."""
    print(f"[INFO] Registering face: {name} from {image_path}")
    
    # Load existing encodings
    known_encodings, known_names = load_existing_encodings()
    
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            print(f"[WARN] No face found in {image_path}, skipping.")
            return False
        
        # Add the first face encoding
        known_encodings.append(encodings[0])
        known_names.append(name)
        
        print(f"[INFO] Successfully registered face for {name}")
        
    except Exception as e:
        print(f"[ERROR] Failed to register face: {e}")
        return False
    
    # Save updated encodings
    data = {
        "encodings": known_encodings,
        "names": known_names
    }
    
    try:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] Updated {ENCODINGS_FILE} with {len(known_encodings)} total faces")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save encodings: {e}")
        return False

if __name__ == "__main__":
    # Only accept API input: employee_name and image_path
    if len(sys.argv) == 3:
        name = sys.argv[1]
        image_path = sys.argv[2]
        success = register_new_face(name, image_path)
        sys.exit(0 if success else 1)
    else:
        print("[ERROR] encodings.py expects exactly 2 arguments: <employee_name> <image_path>")
        print("Usage: python encodings.py \"John Doe\" \"/path/to/image.jpg\"")
        sys.exit(1)
