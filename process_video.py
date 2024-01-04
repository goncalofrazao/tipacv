import scipy.io as sio
import cv2
import numpy as np
import sys
from parse_config_file import parse_config_file

# get command line argument
config_file = sys.argv[1]

# parse the config file
config_dict = parse_config_file(config_file)

video_file = config_dict['videos'][0][0]
features_file = config_dict['keypoints_out'][0][0]

print('Loading video from', video_file)
# Load the video
cap = cv2.VideoCapture(video_file)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
cap.release()

frames = np.array(frames)[::10]

features = []
print('Extracting SIFT features...')
# Create a SIFT object
sift = cv2.SIFT_create()

# Detect SIFT features, with no masks
for i in frames:
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    kps, des = sift.detectAndCompute(gray, None)
    coords = np.array([kp.pt for kp in kps])
    data = np.concatenate((coords, des), axis=1)
    
    features.append(data)

# Generate .mat file with the descriptors
sio.savemat(features_file, {'features': features})
