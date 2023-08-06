import gin
import os
import numpy as np
import open3d as o3d


def rotate_pcd(pcd, angles, center=(0, 0, 0)):
    """Rotates the given point cloud data (pcd) by using given angles for each axis [X, Y, Z].
    Args:
        pcd (object): Open3D Point cloud data.
        angles (list): List containing three angles for the rotation around each axis [X, Y, Z].
        center (tuple): Rotation center.
    Returns:
         Rotated point cloud data.
    """
    # Convert angles
    angles = convert_angles(angles)

    # Build rotational matrix
    rotational_matrix = pcd.get_rotation_matrix_from_xyz((angles[0], angles[1], angles[2]))

    # Rotate using rotational matrix R around center
    return pcd.rotate(rotational_matrix, center=center)


def convert_angles(angles):
    """ Convert angles from degrees to radians.
    Args:
        angles (list): List containing three angles in degrees.
    Returns:
        angles: List containing three angles as multiple from pi.
    """
    for i, angle in enumerate(angles):
        angles[i] = (angle / 180) * np.pi

    return angles


def estimate_vertex_normals(pcd, radius, max_nn):
    """Estimate vertex normals for given pcd.
    Args:
        pcd (object): Open3D Point Cloud Data (PCD).
        radius (float): gin-configurable - Radius around each point, used for vertex normal estimation.
        max_nn (int): gin-configurable - Maximum neighbouring points, used for vertex normal estimation.
    Returns:
        pcd: Open3D Point Cloud Data (PCD) with estimated vertex normals.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # pcd.orient_normals_consistent_tangent_plane(100)

    return pcd


def visualize_keyframe_position(poses_path, sequence, keyframe_number):
    """Visualize given keyframe position as box.
    Args:
        poses_path (str): Path to the saved poses (as .txt).
        sequence (str): Sequence number.
        keyframe_number (int): Keyframe number to visualize position.
    Returns:
        open3D object visualizing the keyframe position.
    """
    # Get poses path
    sequence_poses_path = os.path.join(poses_path, sequence + '.txt')

    # Load sequence poses from .txt file
    poses_data = np.loadtxt(sequence_poses_path, dtype='float32')

    # Select keyframe pose from sequence poses
    extrinsic_data_of_given_keyframe = poses_data[keyframe_number]

    # Extract rotational matrix from extrinsic parameters
    rotational_matrix = np.array([extrinsic_data_of_given_keyframe[0:3].tolist(),
                                  extrinsic_data_of_given_keyframe[4:7].tolist(),
                                  extrinsic_data_of_given_keyframe[8:11].tolist()])

    # Extract translational matrix from extrinsic parameters
    translational_matrix = np.array([extrinsic_data_of_given_keyframe[3],
                                     extrinsic_data_of_given_keyframe[7],
                                     extrinsic_data_of_given_keyframe[11]])

    # Create box model
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.2,
                                                    height=1.0,
                                                    depth=1.0)
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    # Move box model based on given extrinsic parameters
    mesh_box.translate(translational_matrix)
    #mesh_box.transform(rotational_matrix)

    return rotate_pcd(mesh_box)


@gin.configurable
def remove_points_within_radius(pcd_1, pcd_2, radius):
    """Removes points from pcd_1 that are within a given radius of points in pcd_2.
    Args:
        pcd_1 (object): First point cloud data which points should be deleted.
        pcd_2 (object): Second point cloud data which points should be kept.
        radius (float): Radius defining the distance point get deleted between pcd_1 and pcd_2.
    Returns:
        filtered_point_cloud: pcd_1 without the points in the radius to point of pcd_2.
    """
    points_1 = np.asarray(pcd_1.points)
    points_2 = np.asarray(pcd_2.points)

    kd_tree = o3d.geometry.KDTreeFlann()
    kd_tree.set_geometry(pcd_2)

    distances = []
    indices = []

    # Calculate the distance between close points of pcd_1 and pcd_2
    for point in points_1:
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius)
        if len(idx) == 0:
            distances.append(float('inf'))
        else:
            distances.append(np.min(np.linalg.norm(points_2[idx] - point, axis=1)))
        indices.append(idx)

    # Generate mask, whether each point distance lies within the radius or not
    mask = np.array(distances) > radius

    # Filter pcd by generated mask
    filtered_points = points_1[mask]
    filtered_colors = np.asarray(pcd_1.colors)[mask]
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud
