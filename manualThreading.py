import cv2


class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for the main camera
        self.cap = cv2.VideoCapture(self.stream_id)
        if self.cap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        self.grabbed, self.frame = self.cap.read()

        if not self.grabbed:
            print("[Exiting]: Error reading from webcam stream.")
            exit(0)

        self.stopped = False

    def start(self):
