import cv2

CAMERA_SOURCE = "http://192.168.1.5:8080/video"


def parse_camera_source(arg: str):
    """Return int for webcam indices, otherwise assume IP/RTSP/HTTP URL."""
    return int(arg) if arg.isdigit() else arg

def openCam():
    print("[INFO] Starting webcam...")
    camera_source = parse_camera_source(CAMERA_SOURCE)
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    return cap