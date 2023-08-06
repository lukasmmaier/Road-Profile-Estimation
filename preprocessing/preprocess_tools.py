import gin
import open3d as o3d
import numpy as np
import itertools


def crop_pcd(pcd, bounds):
    """Crops given point cloud data (pcd) to bounding box given through bounds.
    Args:
        pcd (object): Open3D Point cloud data.
        bounds (object): Open3D object defining the bounds.
    Returns:
        Cropped point cloud data.
    """
    # Convert to numpy array
    bounds = np.array(bounds, dtype='float32')

    # Create bounding box X, Y, Z:
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

    # Crop the point cloud using the bounding box:
    return pcd.crop(bounding_box)


def paint_uniform_pcd(pcd, color):
    """Print the given point cloud data (pcd) uniformly in a given color.
    Args:
        pcd (object): Open3D Point cloud data.
        color (list): List describing the color value as RGB-color.
    Returns:
        Colord pcd.
    """
    # Paint the point cloud in given color
    return pcd.paint_uniform_color(color)
