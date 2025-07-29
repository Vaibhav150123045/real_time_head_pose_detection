# importing required libraries
import cv2  # OpenCV library
import time  # time library


# opening video capture stream
vcap = cv2.VideoCapture(0)

# Try setting a higher FPS (may not work)
vcap.set(cv2.CAP_PROP_FPS, 500)

if vcap.isOpened() is False:
    print("[Exiting]: Error accessing webcam stream.")
    exit(0)

# processing frames in input stream
num_frames_processed = 0

while True:
    start = time.time()
    grabbed, frame = vcap.read()
    if grabbed is False:
        print('[Exiting] No more frames to read')
        break
    # adding a delay for simulating video processing time
    num_frames_processed += 1
    # displaying frame
    end = time.time()
    total_time = end - start
    fps = 1 / total_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


# printing time elapsed and fps
elapsed = end-start
fps = num_frames_processed/elapsed
print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
# releasing input stream , closing all windows
vcap.release()
cv2.destroyAllWindows()
