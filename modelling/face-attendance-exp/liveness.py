"""
liveness.py
Blink-based liveness detection using Eye Aspect Ratio (EAR).
"""
import numpy as np


class BlinkLiveness:
    """
    Tracks per-person blink-based liveness using Eye Aspect Ratio (EAR).
    Person is considered 'live' once a blink is detected at least once.
    """

    def __init__(self, names, ear_thresh: float = 0.21):
        """
        Initialize liveness tracker.
        
        Args:
            names: List of known person names.
            ear_thresh: Eye Aspect Ratio threshold (below = eye closed).
        """
        self.ear_thresh = ear_thresh
        # per-person state: last_ear + blinked flag
        self.state = {
            name: {"last_ear": 1.0, "blinked": False} for name in names
        }

    @staticmethod
    def _ear(eye_points):
        """
        Compute eye aspect ratio from 6 eye landmarks.
        
        Based on Soukupová & Tereza "Real-Time Eye Blink Detection using
        Facial Landmarks" (CVPR 2016).
        
        Args:
            eye_points: List of 6 (x, y) tuples for eye landmarks.
            
        Returns:
            Eye Aspect Ratio (float).
        """
        if len(eye_points) < 6:
            return 1.0

        p0 = np.array(eye_points[0])
        p1 = np.array(eye_points[1])
        p2 = np.array(eye_points[2])
        p3 = np.array(eye_points[3])
        p4 = np.array(eye_points[4])
        p5 = np.array(eye_points[5])

        A = np.linalg.norm(p1 - p5)
        B = np.linalg.norm(p2 - p4)
        C = np.linalg.norm(p0 - p3)

        if C == 0:
            return 1.0

        ear = (A + B) / (2.0 * C)
        return ear

    def update(self, name: str, landmarks_dict: dict):
        """
        Update liveness state for a recognized person using current landmarks.
        Detects blink as an open -> closed transition in EAR.
        
        Args:
            name: Name of recognized person.
            landmarks_dict: Output of face_recognition.face_landmarks() for one face.
        """
        if name not in self.state:
            return

        if "left_eye" not in landmarks_dict or "right_eye" not in landmarks_dict:
            return

        left_eye = landmarks_dict["left_eye"]
        right_eye = landmarks_dict["right_eye"]

        ear_left = self._ear(left_eye)
        ear_right = self._ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        s = self.state[name]
        was_open = s["last_ear"] > self.ear_thresh
        is_closed = ear < self.ear_thresh

        # open -> closed transition = blink
        if was_open and is_closed:
            s["blinked"] = True
            print(f"[LIVENESS] Detected blink for {name}")

        s["last_ear"] = ear

    def has_blinked(self, name: str) -> bool:
        """
        Check if person has blinked at least once.
        
        Args:
            name: Person name.
            
        Returns:
            True if blink was detected, False otherwise.
        """
        s = self.state.get(name)
        if not s:
            return False
        return s["blinked"]
