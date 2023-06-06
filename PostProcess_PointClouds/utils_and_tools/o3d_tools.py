import gin
import os
import numpy as np
import open3d as o3d

from preprocessing.preprocess_tools import rotate_pcd


@gin.configurable
def estimate_vertex_normals(pcd, radius, max_nn):
    ''' Estimate vertex normal. '''
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # pcd.orient_normals_consistent_tangent_plane(100)

    return pcd


@gin.configurable
def visualize_keyframe_position(poses_path, sequence, keyframe_number, number_of_sub_frames):
    """ Visualize given keyframe position as box. """
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




