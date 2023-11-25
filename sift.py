import cv2
import time

CONTRAST_THRESHOLD = 0.1
EDGE_THRESHOLD = 8
RESPONSE_THRESHOLD = 0.2
FILTERING = False


# Load video
video = cv2.VideoCapture("test.mp4")

# Initialize SIFT detector with increased thresholds
sift = cv2.SIFT_create(
    contrastThreshold=CONTRAST_THRESHOLD, edgeThreshold=EDGE_THRESHOLD
)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if FILTERING:
        keypoints = [kp for kp in keypoints if kp.response > RESPONSE_THRESHOLD]

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
