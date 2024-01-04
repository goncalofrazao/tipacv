import cv2
import numpy as np

def calculate_optical_flow(video_path):
    # Capture video
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Create a video writer for the output video (if you want to save it)
    # Uncomment these two lines to enable saving the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))

    while True:
        # Read next frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Visualize the optical flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = np.zeros_like(frame1)
        mask[..., 1] = 255
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Display the result
        cv2.imshow("Optical Flow", rgb)

        # Write the frame to the output video
        # Uncomment the next line to enable saving the output
        out.write(rgb)

        # Update the previous frame
        prev_gray = gray

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release everything
    cap.release()
    # out.release()  # Uncomment this line if you are saving the output
    cv2.destroyAllWindows()

# Example usage
calculate_optical_flow("tesla_videos/left_repeater_drift.mp4")
