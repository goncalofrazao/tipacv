import cv2
import numpy as np
import open3d as o3d

def extract_first_frame(video_path):
    """ Extract the first frame from a video """
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    for i in range(10):
        _, _ = cap.read()
    sucess2, frame2 = cap.read()
    for i in range(10):
        _, _ = cap.read()
    sucess3, frame3 = cap.read()
    cap.release()
    return [frame, frame2, frame3] if success and sucess2 and sucess3 else None

def find_keypoints_and_descriptors(frame):
    """ Find keypoints and descriptors in a frame using SIFT """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_descriptors(descriptors1, descriptors2):
    """ Match descriptors between two frames """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def reconstruct_3d(matches, keypoints1, keypoints2, camera_matrix):
    """ Reconstruct 3D points from matches """
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    E, _ = cv2.findEssentialMat(points1, points2, camera_matrix)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)

    # For simplicity, assume first camera is at the origin
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # Triangulate points (you might need to adjust this part)
    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = points_3d_homogeneous[:3, :] / points_3d_homogeneous[3, :]

    return points_3d.T

# Paths to your video files
video_paths = ["../tesla_videos/left_repeater.mp4", "../tesla_videos/left_repeater.mp4", "../front.mp4"]

# Load the first frame from each video
frames = extract_first_frame(video_paths[0])

# Camera intrinsics (replace with your values)
camera_matrix = np.array([[1007.4921, 0, 622.32689],
                          [0, 1001.9244, 481.29046],
                          [0, 0, 1]])

# Find keypoints and descriptors in each frame
keypoints_descriptors = [find_keypoints_and_descriptors(frame) for frame in frames]

# Match descriptors between frames (this is simplified for two frames only)
matches = match_descriptors(keypoints_descriptors[0][1], keypoints_descriptors[1][1])

# Reconstruct 3D points (this example uses only the first two frames)
points_3d = reconstruct_3d(matches, keypoints_descriptors[0][0], keypoints_descriptors[1][0], camera_matrix)

# points_3d contains the 3D coordinates of matched points
