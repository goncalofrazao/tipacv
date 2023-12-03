import cv2
import numpy as np
import scipy.io as sio


class FeatureMatcher:
    def __init__(self, video_path, features_path):
        # Load video if path is provided (used for displaying matched frames)
        if video_path is not None:
            self.video = cv2.VideoCapture(video_path)

        self.features = self.get_features(features_path) # List of features

    def get_features(self, features_path):
        """
        Load features from .mat file
        
        Args:
            features_path: path to .mat file containing features
        
        Return:
            features: list of features
        """
        data = sio.loadmat(features_path)
        return [data["features"][0][i] for i in range(len(data["features"][0]))]

    def match_frames(self, frame_index1, frame_index2):
        """
        Match features between 2 frames
        
        Args:
            frame_index1: index of first frame
            frame_index2: index of second frame
        
        Return:
            matched_coords: NumPy array [frame, match, x/y]; shape (2, N, 2)

        """
        # Get features from each frame
        features1 = self.features[frame_index1]
        features2 = self.features[frame_index2]

        # Convert features to OpenCV format
        keypoints1, descriptors1 = self.convert_features(features1)
        keypoints2, descriptors2 = self.convert_features(features2)

        # Match features
        bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bfm.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        return self.get_feature_coordinates(keypoints1, keypoints2, matches)
    
    def match_all_to_ref_frame(self, ref_frame_index):
        """
        Match features between all frames and a reference frame

        Args:
            ref_frame_index: index of reference frame

        Return:
            matched_coords: NumPy array [frame_pair, frame, match, x/y]; shape (N-1, 2, N, 2)

        """
        matches = []
        for i in range(len(self.features)):
            if i == ref_frame_index:
                continue
            matches.append(self.match_frames(ref_frame_index, i))
        
        return np.array(matches)
        


    def get_feature_coordinates(self, keypoints1, keypoints2, matches):
        """
        Get coordinates of matched features
        
        Args:
            keypoints1: keypoints from first frame
            keypoints2: keypoints from second frame
            matches: matches between keypoints
            
        Return:
            matched_coords: NumPy array [frame, match, x/y]; shape (2, N, 2)
        """

        matched_coords = np.zeros((2, len(matches), 2))

        for i, match in enumerate(matches):
            matched_coords[0, i, :] = [
                keypoints1[match.queryIdx].pt[0],
                keypoints1[match.queryIdx].pt[1],
            ]
            matched_coords[1, i, :] = [
                keypoints2[match.trainIdx].pt[0],
                keypoints2[match.trainIdx].pt[1],
            ]

        return matched_coords

    def convert_features(self, features):
        """
        Convert features to OpenCV format

        Args:
            features: list of features

        Return:
            keypoints: list of keypoints
            descriptors: NumPy array of descriptors; shape (N, 128)
        """
        keypoints = []
        descriptors = []

        for feature in features:
            keypoints.append(cv2.KeyPoint(feature[0], feature[1], 1))
            descriptors.append(feature[2:])

        return keypoints, np.float32(descriptors)

    def draw_matches(self, frame1, frame2, matches, points_to_draw=50):
        """
        (Debug function)
        Draw matches between 2 frames

        Args:
            frame1: first frame
            frame2: second frame
            matches: matches between frames
            points_to_draw: number of points to draw
        """
        # Select frames to display
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame1)
        _, frame1 = self.video.read()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame2)
        _, frame2 = self.video.read()

        # Join the 2 frames vertically
        h1, w1, _ = frame1.shape
        h2, w2, _ = frame2.shape
        width = max(w1, w2)
        height = h1 + h2
        matched_frame = np.zeros((height, width, 3), dtype=np.uint8)
        matched_frame[:h1, :w1, :] = frame1
        matched_frame[h1 : h1 + h2, :w2, :] = frame2

        # Draw lines between matched points
        for i in range(min(len(matches[0]), points_to_draw)):
            pt1 = (int(matches[0][i][0]), int(matches[0][i][1]))
            pt2 = (
                int(matches[1][i][0]),
                int(matches[1][i][1]) + h1,
            )  # Shift y-coordinate for frame 2
            cv2.circle(matched_frame, pt1, 2, (0, 255, 0), -1)
            cv2.circle(matched_frame, pt2, 2, (0, 255, 0), -1)
            cv2.line(matched_frame, pt1, pt2, (255, 125, 0), 1)

        # Display the matched frame
        cv2.imshow("Matched Frames", matched_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#* test function 
def test_FeatureMatcher(frame1, frame2):
    video_path = "drone.mp4"
    features_path = "features.mat"
    matcher = FeatureMatcher(video_path, features_path)

    # Get matched coordinates
    matches = matcher.match_frames(frame1, frame2)

    matcher.draw_matches(frame1, frame2, matches)


if __name__ == "__main__":
    test_FeatureMatcher(1, 40)
