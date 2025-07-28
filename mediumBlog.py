# importing required libraries
import cv2  # OpenCV library
import time  # time library

# opening video capture stream
vcap = cv2.VideoCapture(0)
if vcap.isOpened() is False:
    print("[Exiting]: Error accessing webcam stream.")
    exit(0)

# processing frames in input stream
num_frames_processed = 0
start = time.time()
while True:
    grabbed, frame = vcap.read()
    if grabbed is False:
        print('[Exiting] No more frames to read')
        break
    # adding a delay for simulating video processing time
    delay = 0.03  # delay value in seconds
    time.sleep(delay)
    num_frames_processed += 1
    # displaying frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()


# printing time elapsed and fps
elapsed = end-start
fps = num_frames_processed/elapsed
print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
# releasing input stream , closing all windows
vcap.release()
cv2.destroyAllWindows()
