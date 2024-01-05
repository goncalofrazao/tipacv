import cv2
import scipy.io as sio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from feature_matching import FeatureMatcher
import scipy.spatial.transform as st
import sys


class Trajectory:
    def __init__(self, K, features_file=None, features_pairs=None):
        self.K = K
        self.essentials = []
        self.transforms = []
        self.trajectory = []
        self.point_cloud = []
        self.masks = []
        if features_pairs:
            self.features_pairs = features_pairs
        elif features_file:
            self.feature_matcher = FeatureMatcher(None, features_file)
            self.features_pairs = self.feature_matcher.match_consecutive_frames()[1]
            self.features_pairs = self.features_pairs
            print(len(self.features_pairs))

    def get_essentials(self):
        for i, j in self.features_pairs:
            E, mask = cv2.findEssentialMat(i, j, self.K, method=cv2.RANSAC)
            self.masks.append(mask)
            self.essentials.append(E)
        return self.essentials
    
    def get_essentials2(self):
        self.essentials2 = []
        for i in range(len(self.features_pairs) - 1):
            E, _ = cv2.findEssentialMat(self.features_pairs[i][0], self.features_pairs[i + 1][1], self.K, method=cv2.RANSAC)
            self.essentials2.append(E)
        return self.essentials2

    def get_transformations(self):
        for (i, j), E in zip(self.features_pairs, self.essentials):
            _, R, T, _ = cv2.recoverPose(E, i, j)
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3] = T.reshape(3)
            self.transforms.append(H)
        return self.transforms
    
    def get_transformations2(self):
        self.transforms2 = []
        for i in range(len(self.features_pairs) - 1):
            _, R, T, _ = cv2.recoverPose(self.essentials2[i], self.features_pairs[i][0], self.features_pairs[i+1][1])
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3] = T.reshape(3)
            self.transforms2.append(H)
        return self.transforms2
    
    def get_relative_translations(self):
        for i in range(len(self.features_pairs) - 1):
            r = self.transforms[i + 1][:3, :3]
            t1 = r @ self.transforms[i][:3, 3]
            t2 = self.transforms[i + 1][:3, 3]
            t3 = self.transforms2[i][:3, 3]
            print(t1.round(2), t2.round(2), t3.round(2), sep='\n')
            A = np.zeros((3, 3))
            A[:,0] = t1
            A[:,1] = t2
            A[:,2] = -t3
            print(A.round(2))
            _, _, V = np.linalg.svd(A)
            sigmas = V[-1]
            sigmas = sigmas / sigmas[0]
            print(sigmas.round(2), end='\n\n')
            self.transforms[i + 1][:3, 3] = sigmas[1] * t2
            self.transforms2[i][:3, 3] = sigmas[2] * t3

    def get_trajectory(self):
        self.trajectory.append(np.eye(4))
        for H in self.transforms:
            # Update the absolute rotation and translation
            H_i = H @ self.trajectory[-1]
            self.trajectory.append(H_i)

        return self.trajectory

    def get_point_cloud(self):
        for i in range(len(self.trajectory) - 1):
            cameraMatrix1 = self.K @ self.trajectory[i][:3, :]
            cameraMatrix2 = self.K @ self.trajectory[i + 1][:3, :]
            points1 = self.features_pairs[i][0]
            points2 = self.features_pairs[i][1]
            mask = self.masks[i].flatten()
            points1 = points1[mask == 1]
            points2 = points2[mask == 1]

            points = cv2.triangulatePoints(
                cameraMatrix1, cameraMatrix2, points1.T, points2.T
            )
            points = points[:3] / points[3]
            if i == 0:
                self.point_cloud = points.T
            else:
                self.point_cloud = np.vstack((self.point_cloud, points.T))
                # print(np.linalg.norm(points.T - self.point_cloud))
        print(self.point_cloud.shape)

    def plot_frame(self, ax, H, axis_length=1.0):
        # Get the origin and the axes directions
        origin = np.linalg.inv(H) @ np.array([0, 0, 0, 1]).reshape(4, 1)
        x_dir = np.linalg.inv(H) @ np.array([axis_length, 0, 0, 1]).reshape(4, 1)
        y_dir = np.linalg.inv(H) @ np.array([0, axis_length, 0, 1]).reshape(4, 1)
        z_dir = np.linalg.inv(H) @ np.array([0, 0, axis_length, 1]).reshape(4, 1)

        # Plot the axes
        ax.plot(*np.hstack([origin, x_dir])[:3], color="r")
        ax.plot(*np.hstack([origin, y_dir])[:3], color="g")
        ax.plot(*np.hstack([origin, z_dir])[:3], color="b")

    def plot_frames(self):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for H in self.trajectory:
            self.plot_frame(ax, H, axis_length=0.2)

        # Plot the points
        ax.scatter(*self.point_cloud.T, color="k", marker="o")

        # Set labels and plot limits
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        # ax.set_xlim([-5, 5])
        # ax.set_ylim([-5, 5])
        # ax.set_zlim([-5, 5])
        # plt.show()

        # Show the plot
        plt.savefig("point_cloud.png")
        sio.savemat("point_cloud.mat", {"point_cloud": self.point_cloud})

    def plot_trajectory(self):
        # plot 3d points
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x, y, z = [], [], []
        for i in self.trajectory:
            i = np.linalg.inv(i)
            x.append(i[0, 3])
            y.append(i[1, 3])
            z.append(i[2, 3])

        ax.scatter(x, y, c="r", marker="o")
        ax.plot(x, y, c="r")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        # ax.set_xlim3d(-5, 5)
        # ax.set_ylim3d(-5, 5)
        # ax.set_zlim3d(-5, 5)
        # plt.show()
        plt.savefig("trajectory.png")
        sio.savemat("trajectory.mat", {"trajectory": self.trajectory})

    def open3d_plot_pc(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        # change color
        pcd.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd])


def main():
    data = sio.loadmat('new_data.mat')
    K = data['K']
    matches = []
    for i in range(len(data['frames']) - 1):
        matches.append((data['frames'][i], data['frames'][i+1]))
    traj = Trajectory(K, features_pairs=matches)

    # K = np.array([[1007.4921, 0, 622.32689], # left
    #                 [0, 1001.9244, 481.29046],
    #                 [0, 0, 1]])
    # K = np.array([[519.4039, 0, 656.7379], # back
    #                 [0, 518.0534, 451.5029],
    #                 [0, 0, 1]])
    # K = np.array(
    #     [[1600.7216, 0, 601.50012], [0, 1628.0265, 516.2108], [0, 0, 1]]  # front
    # )
    # traj = Trajectory(K, features_file=sys.argv[1])
    traj.get_essentials()
    # traj.get_essentials2()
    traj.get_transformations()
    # traj.get_transformations2()
    # traj.get_relative_translations()
    traj.get_trajectory()
    traj.get_point_cloud()

    traj.plot_frames()
    traj.plot_trajectory()
    # traj.open3d_plot_pc()


if __name__ == "__main__":
    main()
