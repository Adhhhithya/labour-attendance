import os
import face_recognition
import pickle

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

# Loop through all images in known_faces/
for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Name of the person = filename without extension
    name = os.path.splitext(filename)[0]
    path = os.path.join(KNOWN_FACES_DIR, filename)

    print(f"[INFO] Processing {path} as {name}")

    # Load the image
    image = face_recognition.load_image_file(path)

    # Get face encodings (we expect ONE face per image)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"[WARN] No face found in {filename}, skipping.")
        continue

    # Take the first (and only) face
    known_encodings.append(encodings[0])
    known_names.append(name)

print("\n[INFO] Loaded", len(known_encodings), "known faces:")
print(known_names)

# Optionally save them to a file for later use
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Saved encodings to {ENCODINGS_FILE}")
