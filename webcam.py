import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default built-in camera

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Create a window to display the feed
cv2.namedWindow("MacBook Camera", cv2.WINDOW_NORMAL)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Display the resulting frame
    cv2.imshow('MacBook Camera', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
