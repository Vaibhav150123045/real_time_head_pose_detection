import cv2
import time
from manualThreading import WebcamStream

# Initialize the webcam stream
webcam_stream = WebcamStream(0)
webcam_stream.start()

while webcam_stream.stopped is False:
    start = time.time()

    key = cv2.waitKey(1)
    if key == ord('q'):
        webcam_stream.stop()
        exit(0)

    # Read a frame from the webcam stream
    frame = webcam_stream.read()

    if frame is None:
        print("[Exiting]: No more frames to read")
        break

    # Display the frame
    cv2.imshow('Webcam Stream', frame)

    # Calculate and display FPS
    end = time.time()
    total_time = end - start
    fps = 1 / total_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Check for exit key
