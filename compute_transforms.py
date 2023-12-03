import scipy.io as sio
import numpy as np
from parse_config_file import parse_config_file
import sys
import argparse as ap

import cv2
import matplotlib.pyplot as plt
from feature_matching import FeatureMatcher

def homography(destiny, origin):
    """
    Compute the homography matrix for the given points
    
    Args:
        destiny: points from the image to be warped to
        origin: points from the image to be warped 
    
    Return:
        homography: the homography matrix
    """

    # create the matrix A
    A = []
    for ori, dest in zip(origin, destiny):
        # get the coordinates of the origin and destiny points
        x_o, y_o, _ = ori
        x_d, y_d, _ = dest

        # create the rows of A  
        a_x = np.array([-x_d, -y_d, -1, 0, 0, 0, x_o*x_d, x_o*y_d, x_o])
        a_y = np.array([0, 0, 0, -x_d, -y_d, -1, y_o*x_d, y_o*y_d, y_o])

        # append the rows to A
        A.append(a_x)
        A.append(a_y)

    # convert A to numpy array by rows
    A = np.array(A)

    # compute the SVD of A
    U, S, V_t = np.linalg.svd(A)

    # get the last row of V_t
    homography = V_t[-1]

    # #Ax=0 => Eigenvector of A.T@A with smallest eigenvalue
    # eigenvalues, eigenvectors = np.linalg.eig(a.T @ a)
    # H = eigenvectors[:, np.argmin(eigenvalues)].reshape(3, 3)
    # H = H / H[2, 2]

    return homography

def read_file(file):
    """
    Read the input configuration file and return the path to the features file, 
    the feature matching between the map and the frames and the output file
    
    Args:
        file: input file
    
    Return:
        features_file: features file path 
        output_file: output file path
        matchings: feature matching between the map and the frames
    """

    # Initialize the variables
    map_keypoints = []
    frames_keypoints = {}

    # Iterate over the lines of the file
    for line in file:
        # Remove the new line character and any leading or trailing white spaces
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        
        # Split the line into tokens
        tokens = line.split()

        # Check if the line is valid
        if tokens[0] == "keypoints_out":
            # Get the features path and file name
            feature_file = tokens[1]

        elif tokens[0] == "transforms_out":
            # Get the output path and file name
            output_file = tokens[1]

        elif tokens[0] == "pts_in_frame":
            # Get the frame number
            frame_number = int(tokens[1])
            # Get the points
            points = []
            for i in range(2, len(tokens) - 1, 2):
                # Get the point coordinates
                x = tokens[i]
                y = tokens[i + 1]
                # Append the point to the list
                points.append((x, y))
            # Append the points to the dictionary
            frames_keypoints[frame_number] = points
            
        elif tokens[0] == "pts_in_map":
            # Get the points
            points = []
            for i in range(2, len(tokens) - 1, 2):
                # Get the point coordinates
                x = tokens[i]
                y = tokens[i + 1]
                # Append the point to the list
                points.append((x, y))
            # Append the points to the list
            map_keypoints.append(points)
    
    # Get the matchings between the map and the frames
    matchings = np.empty((1, len(map_keypoints)), dtype=object)
    matchings_ind = np.zeros((1, len(map_keypoints)), dtype=int)
    for i, frame in enumerate(frames_keypoints):
        # Get the frame points
        frame_kp = frames_keypoints[frame]
        # Get the map points
        map_kp = map_keypoints[i]
        # Concatenate the points
        points = np.concatenate((frame_kp, map_kp), axis=1)
        # Append the points to the matchings array
        matchings[0][i] = points
        matchings_ind[0, i] = frame

    return feature_file, output_file, (matchings, matchings_ind)

def ransac(data, threshold = 6, n_iterations = 72, n_points = 4):
    parsed_data = []
    for frames in data:
        frames = np.concatenate([frames, np.ones((2, frames.shape[1], 1))], axis=2)
        indexes = [i for i in range(frames.shape[1])]
        np.random.shuffle(indexes)

        features1, features2 = frames[0], frames[1]

        best_inliers = np.array([[], []])
        
        for i in range(n_iterations):
            idx = i * n_points
            x, y = features1[indexes[idx: idx + n_points]], features2[indexes[idx :idx + n_points]]
            h = homography(x, y)
            h = (h / h[-1]).reshape(3, 3)

            y_hat = (h @ features1.T).T
            y_hat = (y_hat / y_hat[:, -1].reshape(-1, 1))

            norm = np.linalg.norm(y_hat - features2, axis=1)
            condition = norm < threshold
            inliers = np.array([features1[condition], features2[condition]])
            
            if inliers.shape[1] > best_inliers.shape[1]:
                best_inliers = inliers
        
        parsed_data.append(best_inliers)
    
    return np.array(parsed_data)
    
def compute_homographies(data):
    homographies = []
    for frame in data:
        h = homography(frame[0], frame[1])
        h /= h[-1]
        homographies.append(h)
    
    return np.array(homographies)

def main():
    # Command line arguments
    # parser = ap.ArgumentParser(description='Compute the transformations between each frame and the map.')
    # parser.add_argument("input_file", type=ap.FileType("r"), help="Configurations file path.")
    # args = vars(parser.parse_args())
    # config_file = args["input_file"]

    # Read the configuration file and get the video file path and output file path
    # feature_file, output_file, matchings = read_file(config_file)
    # if not feature_file or not output_file:
    #     raise Exception("Feature file or output file not specified in configuration file.")
    # print("Feature file:", feature_file)

    config_file = sys.argv[1]
    config_dict = parse_config_file(config_file)
    feature_file = config_dict['keypoints_out'][0][0]
    video_file = config_dict['videos'][0][0]

    fm = FeatureMatcher(video_file, feature_file)
    for i in range(10):
        F1 = i + 1
        F2 = i
        FRAMES = 30
        matches = fm.match_frames(F1, F2)
        fm.draw_matches(F1* FRAMES, F2 * FRAMES, matches, points_to_draw=matches.shape[1])
        data = np.array([matches])
        parsed_data = ransac(data, threshold=6, n_iterations=180, n_points=4)
        fm.draw_matches(F1 * FRAMES, F2 * FRAMES, parsed_data[0], points_to_draw=parsed_data.shape[2])
        homographies = compute_homographies(parsed_data)
        homo = homographies[0].reshape(3, 3)

        fm.video.set(cv2.CAP_PROP_POS_FRAMES, F1 * FRAMES)
        _, im1 = fm.video.read()
        fm.video.set(cv2.CAP_PROP_POS_FRAMES, F2 * FRAMES)
        _, im2 = fm.video.read()
        # generate im2 from homography
        im2_t = cv2.warpPerspective(im1, homo, (im1.shape[1], im1.shape[0]))
        
        # plot im1 and im2
        plt.figure(figsize=(10, 10))

        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
        plt.axis(False)
        plt.subplot(3, 1, 2)
        plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
        plt.axis(False)
        plt.subplot(3, 1, 3)
        plt.imshow(cv2.cvtColor(im2_t, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.axis(False)
        plt.show()

    
    
    

if __name__ == '__main__':
    main()