from threading import Thread
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

        self.stopped = True

        # Thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
                print("[Exiting]: No more frames to read")
                break
        self.cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
