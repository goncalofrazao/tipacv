import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import scipy.spatial.transform as st
import cv2

class SyntheticData:
    def __init__(self, filename, num_of_points=10):
        np.random.seed(102)
        self.filename = filename
        f = 1600
        cx, cy = 600, 600
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        self.rotations = [
            st.Rotation.from_euler('z', 0, degrees=True).as_matrix(),
            st.Rotation.from_euler('z', -90, degrees=True).as_matrix(),
            st.Rotation.from_euler('z', 0, degrees=True).as_matrix(),
            st.Rotation.from_euler('z', 0, degrees=True).as_matrix(),
            st.Rotation.from_euler('z', -90, degrees=True).as_matrix(),
            st.Rotation.from_euler('z', 0, degrees=True).as_matrix()
        ]
        self.translations = [
            np.array([0, 0, 0]),
            np.array([5, 0, 0]),
            np.array([5, 0, 0]),
            np.array([5, 0, 0]),
            np.array([5, 0, 0]),
            np.array([5, 0, 0])
        ]
        # self.world_points = np.random.rand(num_of_points, 3) * 10
        self.world_points = np.array([[0,0,2],
                                      [0,1,2],
                                      [0,2,2],
                                      [0,3,2],
                                      [1,3,2],
                                      [1.5,2.5,2],
                                      [2,2,2],
                                      [1.5,1.75,2],
                                      [1,1.5,2],
                                      [4,0,4],
                                      [4,1,4],
                                      [4,2,4],
                                      [4,3,4],
                                      [7,3,3],
                                      [7.5,2,3],
                                      [8,1,3],
                                      [8.5,0,3],
                                      [9,1,3],
                                      [9.5,2,3],
                                      [10,3,3]
                                      ])
        self.frames = []
        # self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        H = np.eye(4)

        for R, T in zip(self.rotations, self.translations):
            # Update the absolute rotation and translation
            H_i = np.eye(4)
            H_i[:3, :3] = R
            t = -(R @ T)
            H_i[:3, 3] = t
            H = H_i @ H

            # Convert to homogeneous coordinates
            ones = np.ones((self.world_points.shape[0], 1))
            points_homogeneous = np.hstack([self.world_points, ones])
            
            # Camera extrinsic matrix (rotation and translation)
            RT = H[:3, :]

            # Project points
            points_camera = RT @ points_homogeneous.T
            points_image_homogeneous = self.K @ points_camera
            # Normalize to get 2D points
            points_2d = points_image_homogeneous[:2, :] / points_image_homogeneous[2, :]
            self.frames.append(points_2d.T)
        return self.frames
    
    def plot_frame(self, ax, H, axis_length=1.0):
        # Get the origin and the axes directions
        origin = np.linalg.inv(H) @ np.array([0, 0, 0, 1]).reshape(4,1)
        x_dir = np.linalg.inv(H) @ np.array([axis_length, 0, 0, 1]).reshape(4,1)
        y_dir = np.linalg.inv(H) @ np.array([0, axis_length, 0, 1]).reshape(4,1)
        z_dir = np.linalg.inv(H) @ np.array([0, 0, axis_length, 1]).reshape(4,1)

        # Plot the axes
        ax.plot(*np.hstack([origin, x_dir])[:3], color='r')
        ax.plot(*np.hstack([origin, y_dir])[:3], color='g')
        ax.plot(*np.hstack([origin, z_dir])[:3], color='b')
    
    def plot_synthetic_data(self):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        H = np.eye(4)

        # Plot each frame using the absolute transformations
        for R, T in zip(self.rotations, self.translations):
            # Update the absolute rotation and translation
            H_i = np.eye(4)
            H_i[:3, :3] = R
            t = -(R @ T)
            H_i[:3, 3] = t
            H = H_i @ H
            self.plot_frame(ax, H, axis_length=2.0)

        # Plot the world points with labels
        ax.scatter(*self.world_points.T, color='k', marker='o')
        # for i, (x, y, z) in enumerate(self.world_points):
        #     ax.text(x, y, z, str(i), color='k')

        # Set labels and plot limits
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])

        # Show the plot
        plt.show()

    def plot_frames(self):
        num_frames = len(self.frames)
        fig, axs = plt.subplots(1, num_frames, figsize=(5*num_frames, 5))
        
        # Plot each frame with the points and labels
        for i, ax in enumerate(axs):
            ax.scatter(*self.frames[i].T, color='k', marker='o')
            for j, (x, y) in enumerate(self.frames[i]):
                ax.text(x, y, str(j), color='k')
            ax.set_title('Frame {}'.format(i+1))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
        
        plt.show()

    def save_data(self):
        data = {"frames": self.frames, "K": self.K}
        sio.savemat(self.filename, data)

def main():
    data = SyntheticData('new_data.mat', num_of_points=10)
    data.generate_synthetic_data()
    # for i, j in zip(data.rotations, data.translations):
    #     print('R:', st.Rotation.from_matrix(i).as_euler('xyz', degrees=True).round(2))
    #     print('t:', j.round(2))
    data.plot_synthetic_data()
    data.plot_frames()
    data.save_data()

if __name__ == '__main__':
    main()
