import cv2
import time


# Load video
video = cv2.VideoCapture("test.mp4")

# Initialize SIFT detector with increased thresholds
sift = cv2.SIFT_create()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints for visualization
    img = cv2.drawKeypoints(frame, keypoints, None)

    # Display the frame
    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.1)

# Release resources
video.release()
cv2.destroyAllWindows()
