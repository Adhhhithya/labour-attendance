import cv2
import pickle
import numpy as np

from arcface_model import load_arcface_model
from attendance import AttendanceLogger

ENCODINGS_FILE = "arcface_encodings.pkl"
MATCH_THRESHOLD = 0.50
CAMERA_SOURCE = "http://192.168.1.3:8080/video"

def parse_camera_source(arg: str):
    """Return int for webcam indices, otherwise assume IP/RTSP/HTTP URL."""
    return int(arg) if arg.isdigit() else arg

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("[INFO] Loading ArcFace model...")
    model = load_arcface_model()

    print("[INFO] Loading enrolled identities...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    names = data["names"]
    embeddings = data["embeddings"]

    attendance = AttendanceLogger()
    camera_source = parse_camera_source(CAMERA_SOURCE)
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = model.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            emb = face.embedding
            landmarks = face.landmark_2d_106

            best_name = "Unknown"
            best_score = 0.0

            for name, known_emb in zip(names, embeddings):
                score = cosine_similarity(emb, known_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score < MATCH_THRESHOLD:
                best_name = "Unknown"

            # Draw UI
            if best_name == "Unknown":
                label = "Unknown"
                color = (0, 0, 255)
            else:
                label = best_name
                color = (0, 255, 0)
                attendance.mark(best_name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("ArcFace Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
