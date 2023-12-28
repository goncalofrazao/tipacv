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

# Load the video
cap = cv2.VideoCapture(video_file)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
cap.release()

# frames = np.array(frames)[0:-1:30]

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

    indices = np.random.choice(data.shape[0], size=200, replace=False)
    selected_data = data[indices]

    # plot only the selected keypoints
    # for j in selected_data:
    #     cv2.circle(i, (int(j[0]), int(j[1])), 2, (0, 0, 255), -1)
    # cv2.imshow('frame', i)
    # cv2.waitKey(0)

    features.append(selected_data)

# Generate .mat file with the descriptors
sio.savemat(features_file, {'features': features})
