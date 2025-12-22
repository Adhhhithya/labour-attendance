import cv2
import cv2
import numpy as np
from arcface_model import load_arcface_model
from webcam_conn import openCam

MATCH_THRESHOLD = 0.60


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def open_camera_with_fallback():
    cap = openCam()
    if cap is not None and cap.isOpened():
        return cap
    for idx in [0, 1, 2]:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            print(f"[INFO] Fallback to local camera index {idx}")
            return c
    return None


def make_faiss_searcher():
    from faiss_utils import init_faiss, INDEX_PATH
    import os
    index, metadata = init_faiss()
    if not os.path.exists(INDEX_PATH) or index.ntotal == 0 or len(metadata) == 0:
        raise RuntimeError("No enrollments found. Run arcface_enroll.py first.")
    return index, metadata


def search_faiss(index, metadata, query_embedding, top_k=1):
    q = _normalize(query_embedding).reshape(1, -1).astype(np.float32)
    scores, ids = index.search(q, top_k)
    best_id = int(ids[0][0])
    best_score = float(scores[0][0])
    name = metadata.get(best_id, "Unknown")
    return name, best_score


def main():
    print("[INFO] Loading ArcFace model...")
    model = load_arcface_model()

    print("[INFO] Preparing FAISS searcher...")
    try:
        index, metadata = make_faiss_searcher()
        print(f"[INFO] FAISS entries: {index.ntotal}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("[INFO] Opening camera...")
    cap = open_camera_with_fallback()
    if cap is None:
        print("[ERROR] Could not open any camera. Check connections.")
        return

    print("[INFO] Starting recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = model.get(frame)
        if not faces:
            try:
                cv2.imshow("ArcFace Recognition", frame)
            except Exception:
                pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name, score = search_faiss(index, metadata, face.embedding)
            if score < MATCH_THRESHOLD:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f"Recognized: {name}, score={score:.2f}")

        try:
            cv2.imshow("ArcFace Recognition", frame)
        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("[INFO] Recognition stopped.")


if __name__ == "__main__":
    main()
