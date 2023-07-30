import gin
import os
from absl import app, flags
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def rotate_pcd(pcd, angles, center=(0,0,0)):
    ''' Rotates the given point cloud data (pcd) by using given angles for each axis [X, Y, Z]. '''
    # Convert angles
    angles = convert_angles(angles)

    # Build rotational matrix
    rotational_matrix = pcd.get_rotation_matrix_from_xyz((angles[0], angles[1], angles[2]))

    # Rotate using rotational matrix R around center
    return pcd.rotate(rotational_matrix, center=center)


def convert_angles(angles):
    ''' Convert angles from degrees to radians. '''
    for i, angle in enumerate(angles):
        angles[i] = (angle / 180) * np.pi

    return angles


def process_for_evaluation(road_profile_reconstruction, pcd, angles, center, height_difference=0.30):

    road_profile_reconstruction = rotate_pcd(road_profile_reconstruction, angles, center)
    pcd = rotate_pcd(pcd, angles, center)

    points = np.asarray(road_profile_reconstruction.points)

    #mean_height = np.mean(points[:, 1])
    mean_height = np.median(points[:, 1])
    #min_height = np.min(points[:, 1])
    #max_height = np.max(points[:, 1])
    min_height = mean_height - height_difference/2
    max_height = mean_height + height_difference/2

    normalized_height = (points[:, 1] - min_height) / (max_height - min_height)
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_height)

    colors = colors[:, 0:3]

    road_profile_reconstruction.colors = o3d.utility.Vector3dVector(colors)

    return road_profile_reconstruction, pcd







