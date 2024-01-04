import cv2
import numpy as np

def calculate_optical_flow_and_plot(video_path, output_path):
    # Capture video
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        # Read the next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Specify the maximum number of corners to detect
        max_corners = 100

        # Calculate optical flow
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7, mask=None)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        # Calculate the average flow vector
        valid_points = next_pts[status == 1]
        valid_prev_points = prev_pts[status == 1]
        flow_vectors = valid_points - valid_prev_points
        average_flow = np.mean(np.linalg.norm(flow_vectors, axis=1))

        # Overlay the instantaneous velocity on the frame
        velocity_text = f"Velocity: {average_flow:.2f} px/frame"
        cv2.putText(frame2, velocity_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame2)

        # Update previous frame
        prev_gray = gray.copy()

    cap.release()
    out.release()

# Replace with the actual paths of your input and output video files
calculate_optical_flow_and_plot('tesla_videos/front_tecas.mp4', 'o2.mp4')
