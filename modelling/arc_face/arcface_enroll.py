import cv2
import sys
import numpy as np
from arcface_model import load_arcface_model
from faiss_utils import init_faiss, save_faiss, add_embedding
from webcam_conn import openCam



def enroll(name,frame):
    print("[INFO] Loading ArcFace model...")
    model = load_arcface_model()

    index, metadata = init_faiss()
    
    # Start ID from the next available
    next_id = max(metadata.keys()) + 1 if metadata else 0

    print(f"[INFO] Enrolling {name}")
    faces = model.get(frame)

    if len(faces) == 0:
        print(f"[WARN] No face found in frame for {name}, skipping enrollment.")
        return

    # Take the most confident face
    face = max(faces, key=lambda f: f.det_score)

    # Add embedding and show index size change
    before = index.ntotal
    add_embedding(index, metadata, next_id, face.embedding, name)
    next_id += 1

    print(f" -> embedding shape: {face.embedding.shape}")
    print(f"[INFO] Index size: {before} -> {index.ntotal}")

    save_faiss(index, metadata)
    from faiss_utils import INDEX_PATH, META_PATH
    print(f"[INFO] FAISS index saved: {INDEX_PATH}")
    print(f"[INFO] Metadata saved: {META_PATH}")

    print(f"\n[INFO] Enrollment complete")
    print(f"[INFO] Saved {len(metadata)} identities to FAISS vector DB")


if __name__ == "__main__":
    cap = openCam()

    name = sys.argv[1] if len(sys.argv) > 1 else "Unknown"

    if cap is None or not hasattr(cap, "isOpened"):
        print("Camera capture failed")
        sys.exit(1)

    captured_frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera capture failed")
                break

            cv2.imshow("ArcFace Enrollment - Press 'E' to enroll, 'Q' to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                captured_frame = frame.copy()
                break
            elif key == ord('q'):
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if captured_frame is not None:
        enroll(name, captured_frame)
        print(f"Face enrolled successfully for {name}")
        sys.exit(0)
    else:
        print("Enrollment cancelled or no frame captured.")
        sys.exit(1)

