import time
import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default built-in camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Create a window to display the feed
cv2.namedWindow("MacBook Camera", cv2.WINDOW_NORMAL)
fps = 0
while cap.isOpened():
    # Capture frame-by-frame
    start = time.time()

    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('MacBook Camera', frame)

    end = time.time()
    total_time = end - start
    fps = 1 / total_time

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything when done
cap.release()
