import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_features(frame):
    # Use ORB to detect and extract features
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    desc1 = np.float32(desc1)
    desc2 = np.float32(desc2)
    # Match features using FLANN matcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.match(desc1, desc2)
    return matches

def estimate_motion(match, kp1, kp2, camera_matrix):
    # Estimate motion from matched features
    points1 = np.float32([kp1[m.queryIdx].pt for m in match])
    points2 = np.float32([kp2[m.trainIdx].pt for m in match])

    # Calculate Essential Matrix
    E, _ = cv2.findEssentialMat(points1, points2, camera_matrix)

    # Decompose Essential Matrix into rotation and translation
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
    return R, t

# Initialize video source
cap = cv2.VideoCapture('tesla_videos/back.mp4')

# Assuming a simple camera model, change this with your camera parameters
focal_length1 = 519.4039
focal_length2 = 518.0534
center = (656.7379, 451.5029)
camera_matrix = np.array([[focal_length1, 0, center[0]], [0, focal_length2, center[1]], [0, 0, 1]], dtype=float)

# Read first frame
ret, prev_frame = cap.read()
prev_keypoints, prev_descriptors = extract_features(prev_frame)

trajectory = np.zeros((3,))  # x, y, z trajectory
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

i = 0
while True:
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, descriptors = extract_features(frame)
    matches = match_features(prev_descriptors, descriptors)
    R, t = estimate_motion(matches, prev_keypoints, keypoints, camera_matrix)

    # Update trajectory (simple approximation)
    trajectory += t.ravel()

    # Plotting
    # ax.scatter(trajectory[0], trajectory[1], trajectory[2])
    if i % 10 == 0:
        ax.scatter(-trajectory[0], trajectory[1])
        plt.pause(0.05)

    prev_frame = frame
    prev_keypoints = keypoints
    prev_descriptors = descriptors

cap.release()
cv2.destroyAllWindows()
