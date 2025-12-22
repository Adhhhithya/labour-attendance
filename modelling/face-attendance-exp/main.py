"""
main.py
Face attendance system with liveness detection (eye blink).

Workflow:
1. Load pre-computed face encodings + known names.
2. Initialize liveness tracker + attendance logger.
3. Capture video frames from webcam.
4. Detect faces, extract encodings + landmarks.
5. Compare against known faces; update liveness state.
6. Display face bounding boxes (color-coded by liveness status).
7. Mark attendance only for recognized + live faces (blink detected).
"""
import cv2
import face_recognition
import numpy as np

from enrollment import load_encodings
from liveness import BlinkLiveness
from attendance import AttendanceLogger


# Set this to your phone's IP Webcam stream URL (or an int index for a USB cam).
# Example: "http://192.168.1.6:8080/video" for the IP Webcam app default video feed.
CAMERA_SOURCE = "http://192.168.1.3:8080/video"


def parse_camera_source(arg: str):
    """Return int for webcam indices, otherwise assume IP/RTSP/HTTP URL."""
    return int(arg) if arg.isdigit() else arg


def main():
    camera_source = parse_camera_source(CAMERA_SOURCE)

    # 1. Load known encodings + names
    print("[INFO] Loading known face encodings...")
    known_encodings, known_names = load_encodings()
    print(f"[INFO] Loaded {len(known_encodings)} known faces: {known_names}")

    # 2. Initialize liveness tracker + attendance logger
    liveness = BlinkLiveness(known_names, ear_thresh=0.21)
    attendance_logger = AttendanceLogger(path="attendance.csv")

    # 3. Start webcam
    print(f"[INFO] Starting camera source: {camera_source}")
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check the URL/index and network.")
        return

    print("[INFO] Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Resize for speed, convert to RGB
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # 4. Detect faces + compute encodings + extract landmarks
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

        face_names = []

        for i, face_encoding in enumerate(face_encodings):
            # Compare against known faces
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=0.5
            )
            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )

            name = "Unknown"
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name = known_names[best_idx]

            face_names.append(name)

            # Update liveness state for recognized faces
            if name != "Unknown" and i < len(landmarks_list):
                liveness.update(name, landmarks_list[i])

        # 5. Draw bounding boxes + labels; mark attendance if live
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates back to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            is_live = name != "Unknown" and liveness.has_blinked(name)

            # Choose color + label based on recognition + liveness status
            if name == "Unknown":
                color = (0, 0, 255)      # red -> unknown
                label = "Unknown"
            elif not is_live:
                color = (0, 255, 255)    # yellow -> recognized but not live yet
                label = f"{name} (blink)"
            else:
                color = (0, 255, 0)      # green -> live & recognized
                label = name

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, label, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

            # Mark attendance only if recognized + live
            if name != "Unknown" and is_live:
                attendance_logger.mark_if_live_and_not_marked(name)

        cv2.imshow("Face Attendance with Liveness", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed. Goodbye!")


if __name__ == "__main__":
    main()
