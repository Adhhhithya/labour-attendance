# liveness.py
import numpy as np


class BlinkLiveness:
    def __init__(self, names, ear_thresh=0.21):
        self.ear_thresh = ear_thresh
        self.state = {
            name: {"last_ear": 1.0, "blinked": False}
            for name in names
        }

    @staticmethod
    def _ear(eye):
        if len(eye) < 6:
            return 1.0

        p0, p1, p2, p3, p4, p5 = [np.array(p) for p in eye]
        A = np.linalg.norm(p1 - p5)
        B = np.linalg.norm(p2 - p4)
        C = np.linalg.norm(p0 - p3)

        return (A + B) / (2.0 * C) if C != 0 else 1.0

    def update(self, name, landmarks):
        if name not in self.state:
            return

        if landmarks is None:
            return

        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return

        ear_left = self._ear(landmarks["left_eye"])
        ear_right = self._ear(landmarks["right_eye"])
        ear = (ear_left + ear_right) / 2.0

        s = self.state[name]
        was_open = s["last_ear"] > self.ear_thresh
        is_closed = ear < self.ear_thresh

        if was_open and is_closed:
            s["blinked"] = True
            print(f"[LIVENESS] Blink detected for {name}")

        s["last_ear"] = ear

    def is_live(self, name):
        return name in self.state and self.state[name]["blinked"]
