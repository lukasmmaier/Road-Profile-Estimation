import gin
from absl import app, flags
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils_and_tools.o3d_tools import rotate_pcd


@gin.configurable()
def process_for_evaluation(road_profile_reconstruction, pcd, angles, center, height_difference):
    """Rotate and colorize the pcd for better evaluation.
    Args:
        road_profile_reconstruction (object): Open3D point cloud data (=post processing output).
        pcd (object): Open3D point cloud data (=MonoRec output).
        angles (list): List containing three angles for the rotation around each axis [X, Y, Z].
        center (tuple): Rotation center.
        height_difference (float): Height difference used to colorize the Y-Axis information.
    Returns:
        road_profile_reconstruction: Colored and rotated Open3D point cloud data
        pcd: Rotated Open3D point cloud data
    """
    # Rotate pcd or road_profile_reconstruction if desired
    road_profile_reconstruction = rotate_pcd(road_profile_reconstruction, angles, center)
    pcd = rotate_pcd(pcd, angles, center)

    points = np.asarray(road_profile_reconstruction.points)

    # Calculate boundaries for colorization
    mean_height = np.median(points[:, 1])
    min_height = mean_height - height_difference/2
    max_height = mean_height + height_difference/2

    # Colorize road_profile_reconstruction
    normalized_height = (points[:, 1] - min_height) / (max_height - min_height)
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_height)
    colors = colors[:, 0:3]
    road_profile_reconstruction.colors = o3d.utility.Vector3dVector(colors)

    return road_profile_reconstruction, pcd
