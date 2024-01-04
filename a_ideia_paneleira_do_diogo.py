import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.io as sio
import numpy as np

# Load video
video_path = "tesla_videos/front_tecas.mp4"
cap = cv2.VideoCapture(video_path)

# Load trajectory
trajectory_path = "a_ideia_paneleira_do_diogo.mat"
data = sio.loadmat(trajectory_path)
trajectory = data["trajectory"]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Add your 3D plot code here

i = 0
# create interactive plot
plt.ion()
plt.show()
# Display video and 3D plot
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display video frame
    cv2.imshow('Video', frame)

    # Update 3D plot
    if i % 10 == 0:
        H = trajectory[i//10]
        H = np.linalg.inv(H)
        x, y, z = H[0, 3], H[1, 3], H[2, 3]
        ax.scatter(x, y, z, c='r', marker='o')

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1
    # time.sleep(1/60)
    

cap.release()
cv2.destroyAllWindows()
