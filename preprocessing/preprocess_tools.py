import gin
import open3d as o3d
import numpy as np
import itertools


@gin.configurable
def rotate_pcd(pcd, angles):
    ''' Rotates the given point cloud data (pcd) by using given angles for each axis [X, Y, Z]. '''
    # Convert angles
    angles = convert_angles(angles)

    # Build rotational matrix
    rotational_matrix = pcd.get_rotation_matrix_from_xyz((angles[0], angles[1], angles[2]))

    # Rotate using rotational matrix R around center
    return pcd.rotate(rotational_matrix, center=(0, 0, 0))


def convert_angles(angles):
    ''' Convert angles from degrees to radians. '''
    for i, angle in enumerate(angles):
        angles[i] = (angle / 180) * np.pi

    return angles


@gin.configurable
def crop_pcd(pcd, bounds):
    ''' Crops given point cloud data (pcd) to bounding box given through bounds. '''
    # Convert to numpy array
    bounds = np.array(bounds, dtype='float32')

    # Create bounding box X, Y, Z:
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

    # Crop the point cloud using the bounding box:
    return pcd.crop(bounding_box)


@gin.configurable
def paint_uniform_pcd(pcd, color):
    ''' Print the given point cloud data (pcd) uniformly in a given color. '''
    # Paint the point cloud in given color
    return pcd.paint_uniform_color(color)
