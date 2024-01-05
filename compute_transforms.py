import scipy.io as sio
import numpy as np
from parse_config_file import parse_config_file
import sys
import argparse as ap

import cv2
import matplotlib.pyplot as plt
from feature_matching import FeatureMatcher

class Homography:
    def __init__(self, config_file_name):
        self.config_file = parse_config_file(config_file_name)
        self.feature_file = self.config_file['keypoints_out'][0][0]
        self.video_file = self.config_file['videos'][0][0]
        self.transforms_file = self.config_file['transforms_out'][0][0]
        
        self.map_features = self.config_file['pts_in_map'][0][1:]
        self.map_features = [float(x) for x in self.map_features]
        self.ref_frame = int(self.config_file['pts_in_frame'][0][0])
        self.ref_frame_features = self.config_file['pts_in_frame'][0][1:]
        self.ref_frame_features = [float(x) for x in self.ref_frame_features]

        self.img_height = int(self.config_file['imagesize'][0][0])
        self.img_width = int(self.config_file['imagesize'][0][1])
        
        mode = self.config_file['transforms'][0][1]

        print("Feature file:", self.feature_file)
        print("Video file:", self.video_file)
        print("Transforms file:", self.transforms_file)
        print("Map features:", self.map_features)
        print("Reference frame:", self.ref_frame)
        print("Reference frame features:", self.ref_frame_features)
        print("Image height:", self.img_height)
        print("Image width:", self.img_width)
        print("Mode:", mode)

        self.reshape_features()

        self.fm = FeatureMatcher(self.video_file, self.feature_file)
        self.indexes, self.matches = self.fm.match_consecutive_frames()
        # self.fm.draw_matches(0, 10, self.matches[0], points_to_draw=self.matches[0].shape[1])
        self.matches = self.ransac(self.matches, threshold=6, n_iterations=220, n_points=4)
        # self.fm.draw_matches(0, 10, self.matches[0], points_to_draw=self.matches[0].shape[1])
        self.homographies = self.compute_homographies(self.matches)

        if mode == 'all':
            self.indexes, self.homographies = self.all_homographies()
        elif mode == 'map':
            self.indexes, self.homographies = self.direct_or_compose()
            self.homographies = self.all_to_map()

        self.save_homographies()

    def all_to_map(self):
        ref_to_map = self.homography(self.map_features, self.ref_frame_features)
        ref_to_map /= ref_to_map[-1]
        ref_to_map = ref_to_map.reshape(3,3)
        homographies = []
        for h in self.homographies:
            H = h.reshape(3,3) @ ref_to_map
            H /= H[-1]
            homographies.append(H.reshape(9,))

        return homographies

    def reshape_features(self):
        self.map_features = np.array(self.map_features).reshape(-1, 2)
        self.ref_frame_features = np.array(self.ref_frame_features).reshape(-1, 2)
        # to homogeneous coordinates
        self.map_features = np.concatenate([self.map_features, np.ones((self.map_features.shape[0], 1))], axis=1)
        self.ref_frame_features = np.concatenate([self.ref_frame_features, np.ones((self.ref_frame_features.shape[0], 1))], axis=1)

    def superimpose_percentage(self, homography):
        points = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))
        points = np.array(points).reshape(2, -1)
        points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)
        points = homography @ points
        points = (points / points[-1]).astype(int)
        points = points[:2]
        points = points.reshape(2, self.img_height, self.img_width)
        points = np.moveaxis(points, 0, -1)
        points = np.moveaxis(points, 0, 0)
        points = np.moveaxis(points, 1, 0)
        points = np.moveaxis(points, 2, 0)
        points = points.reshape(2, -1)
        points = points.T
        points = points[(points[:, 0] >= 0) & (points[:, 0] < self.img_width) & (points[:, 1] >= 0) & (points[:, 1] < self.img_height)]

        # print("-> ", points.shape[0] / (self.img_height * self.img_width))
        return points.shape[0] / (self.img_height * self.img_width)

    def direct_or_compose(self):
        H = [np.eye(3).reshape(9,)]
        indexes = [(0, 0)]
        H_i = np.eye(3)

        H_i = np.eye(3)

        for i, h in enumerate(self.homographies):
            H_i = h.reshape(3,3) @ H_i
            indexes.append((0, i + 1))
            if self.superimpose_percentage(H_i) > 0.55:
                matches = [self.fm.match_frames(0, i + 1)]
                matches = self.ransac(matches, threshold=6, n_iterations=220, n_points=4)
                H_i = self.compute_homographies(matches)[0]
                H.append(H_i)
                H_i = H_i.reshape(3,3)
            else:
                H.append(H_i.reshape(9,))

        return indexes, np.array(H)

    def homography(self, destiny, origin):
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
        if A.shape[0] < 8:
            print("Not enough points to compute homography")
            return np.eye(3)
        # compute the SVD of A
        U, S, V_t = np.linalg.svd(A)

        # get the last row of V_t
        homography = V_t[-1]

        return homography

    def ransac(self, matches, threshold = 6, n_iterations = 72, n_points = 4):
        parsed_data = []
        for frames in matches:
            frames = np.concatenate([frames, np.ones((2, frames.shape[1], 1))], axis=2)
            indexes = [i for i in range(frames.shape[1])]
            np.random.shuffle(indexes)

            features1, features2 = frames[0], frames[1]

            best_inliers = np.array([[], []])
            
            for i in range(n_iterations):
                idx = i * n_points
                x, y = features1[indexes[idx: idx + n_points]], features2[indexes[idx :idx + n_points]]
                h = self.homography(x, y)
                h = (h / h[-1]).reshape(3, 3)

                y_hat = (h @ features1.T).T
                y_hat = (y_hat / y_hat[:, -1].reshape(-1, 1))

                norm = np.linalg.norm(y_hat - features2, axis=1)
                condition = norm < threshold
                inliers = np.array([features1[condition], features2[condition]])
                
                if inliers.shape[1] > best_inliers.shape[1]:
                    best_inliers = inliers

            parsed_data.append(best_inliers)
        
        return parsed_data
    
    def compute_homographies(self, matches):
        homographies = []
        for frame in matches:
            h = self.homography(frame[0], frame[1])
            h /= h[-1]
            homographies.append(h)
        
        return np.array(homographies)

    def all_homographies(self):
        homographies = []
        indexes = []

        for i in range(len(self.homographies)):
            H = np.eye(3)
            for j in range(i + 1, len(self.matches)):
                indexes.append((j, i))
                H = self.homographies[j - 1] @ H
                homographies.append(H.reshape(9,))
        
        return indexes, np.array(homographies)

    def save_homographies(self):
        homographies = np.concatenate([np.array(self.indexes), self.homographies], axis=1)
        sio.savemat(self.transforms_file, {'H': homographies})

def main():
    homo = Homography(sys.argv[1])
    

    FRAMES = 10
    for i,(a,b), (c,d) in zip(range(len(homo.homographies)), homo.indexes, homo.indexes2):
        # print("Homography", i,"(", a - c, ",", b - d, ")", ":", homo.homographies[i].round(2))
        print(a, b, c, d)
        # fm.draw_matches(a * FRAMES, b * FRAMES, matches[i], points_to_draw=matches[i].shape[1])
        # homo.fm.draw_matches(a * FRAMES, b * FRAMES, homo.matches[i])
        
        homo.fm.video.set(cv2.CAP_PROP_POS_FRAMES, a * FRAMES)
        _, im1 = homo.fm.video.read()
        homo.fm.video.set(cv2.CAP_PROP_POS_FRAMES, b * FRAMES)
        _, im2 = homo.fm.video.read()
        
        im2_t = cv2.warpPerspective(im1, homo.homographies[i].reshape(3,3), (im1.shape[1], im1.shape[0]))
        im2_superimposed = cv2.addWeighted(im2, 0.5, im2_t, 0.5, 0)
        
        cv2.imshow("direct", im2_superimposed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
