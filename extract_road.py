import gin
import os
import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

from estimate_road_profile import slice_in_given_dimension
from preprocessing.evaluation_processing import rotate_pcd

@gin.configurable
def get_poses(poses_path, sequence, start_frame, end_frame, long_plot_start_frame):
    """Gets the extrinsic data between the selected start/end frame and returns it as a list"""
    # Get poses path
    sequence_poses_path = os.path.join(poses_path, sequence + '.txt')

    # Load sequence poses from .txt file
    poses_data = np.loadtxt(sequence_poses_path, dtype='float32')

    # Select keyframe poses from given start, end
    extrinsic_data_list = poses_data[start_frame:end_frame+1]

    if (long_plot_start_frame >= start_frame) and (long_plot_start_frame <= end_frame):
        long_plot_start_marker = long_plot_start_frame-start_frame
    else:
        long_plot_start_marker = -1

    return extrinsic_data_list, long_plot_start_marker


def select_points_in_volume(pcd, corner_points):
    """
    Selects points within the volume defined by four given points.

    Arguments:
    - points: numpy array of shape (N, 3) representing the 3D points of the point cloud
    - p1, p2, p3, p4: four points defining the volume

    Returns:
    - selected_points: numpy array of shape (M, 3) containing the selected points within the volume
    """

    bounding_box = corner_points.astype("float64")

    points = o3d.utility.Vector3dVector(bounding_box)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    point_cloud_crop = pcd.crop(oriented_bounding_box)

    selected_points = np.asarray(point_cloud_crop.points)

    # Delete points above streets surface
    # Get min/max height values for selected scene
    #min_y = min(selected_points[:, 1])
    #max_y = min_y + 5

    if not len(selected_points) == 0:
        max_y = max(selected_points[:, 1])
        min_y = max_y - 1

        selected_points = selected_points[
            (selected_points[:, 1] >= min_y) &
            (selected_points[:, 1] <= max_y)
        ]

    oriented_bounding_box.color = [1, 0, 0]

    return selected_points, oriented_bounding_box, point_cloud_crop


def paint_pcd_height(extracted_road_pcd, corner_points):
    bounding_box = corner_points.astype("float64")

    points = o3d.utility.Vector3dVector(bounding_box)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    point_cloud_crop = extracted_road_pcd.crop(oriented_bounding_box)

    #Delete already used points
    dists = np.asarray(extracted_road_pcd.compute_point_cloud_distance(point_cloud_crop))
    indices = np.where(dists > 0.00001)[0]
    pcd_cropped_inv = extracted_road_pcd.select_by_index(indices)

    selected_points = np.asarray(point_cloud_crop.points)

    mean_height = np.mean(selected_points[:, 1])

    min_height = np.min(selected_points[:, 1])
    max_height = np.max(selected_points[:, 1])
    #min_height = mean_height - 0.5
    #max_height = mean_height + 0.5

    normalized_height = (selected_points[:, 1] - min_height) / (max_height - min_height)
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_height)

    colors = colors[:, 0:3]

    return colors, pcd_cropped_inv


def decompose_extrinsic_matrix(extrinsic_matrix):

    extrinsic_matrix = extrinsic_matrix.reshape(3, 4)

    # Extract the rotation matrix and translation vector from the extrinsic parameters
    rotation_matrix = extrinsic_matrix[:, :3]
    translation_vector = extrinsic_matrix[:, 3]

    return rotation_matrix, translation_vector


def transform_points_to_world_coords(extrinsic_matrix, roi_param_r=3, roi_param_l=3, eval_dist_rcs=0.45,
                                     eval_dist_lrp=17, eval_pos_lrp=0):
    """
    Transform points from camera coordinates to world coordinates using extrinsic parameters.

    Args:
    - points: numpy array of shape (2, 3) representing the points in camera coordinates
    - extrinsic_params: numpy array of shape (3, 4) representing the extrinsic parameters

    Returns:
    - transformed_points: numpy array of shape (2, 3) containing the points in world coordinates

        points = np.array([[roi_param, 5, 5.2]
                       [roi_param, -5, 5.2],
                       [-roi_param, 5, 5.2],
                       [-roi_param, -5, 5.2],
                       [0, 0, 0],
                       [2, 0, 0],
                       [-2, 0, 0],
                       [roi_param, 5, 5.3],
                       [roi_param, -5, 5.3],
                       [-roi_param, 5, 5.3],
                       [-roi_param, -5, 5.3]
                       ])

    """

    rotation_matrix, translation_vector = decompose_extrinsic_matrix(extrinsic_matrix)

    points = np.array([[roi_param_r, 5, 0],
                       [roi_param_r, -5, 0],
                       [-roi_param_l, 5, 0],
                       [-roi_param_l, -5, 0],

                       [0, 0, 0],
                       [2, 0, 0],
                       [-2, 0, 0],

                       [roi_param_r, 5, eval_dist_rcs],
                       [roi_param_r, -5, eval_dist_rcs],
                       [-roi_param_l, 5, eval_dist_rcs],
                       [-roi_param_l, -5, eval_dist_rcs],

                       [roi_param_r, 5, eval_dist_rcs+0.10],
                       [roi_param_r, -5, eval_dist_rcs+0.10],
                       [-roi_param_l, 5, eval_dist_rcs+0.10],
                       [-roi_param_l, -5, eval_dist_rcs+0.10],

                       [eval_pos_lrp, 5, 0.3],
                       [eval_pos_lrp, -5, 0.3],
                       [eval_pos_lrp+0.1, 5, 0.3],
                       [eval_pos_lrp+0.1, -5, 0.3],

                       [eval_pos_lrp, 5, eval_dist_lrp],
                       [eval_pos_lrp, -5, eval_dist_lrp],
                       [eval_pos_lrp+0.1, 5, eval_dist_lrp],
                       [eval_pos_lrp+0.1, -5, eval_dist_lrp]
                       ])

    # Transform points to world coordinates
    transformed_points = np.dot(rotation_matrix, points.T) + translation_vector.reshape(-1, 1)

    return transformed_points.T

"""
def extract_road_section(point_cloud, pose_1, pose_2):

    p1_rotation_matrix, p1_translation_vector = decompose_extrinsic_matrix(pose_1)
    p2_rotation_matrix, p2_translation_vector = decompose_extrinsic_matrix(pose_2)

    rotated_point_cloud = point_cloud.rotate(np.transpose(p1_rotation_matrix), center=p1_translation_vector)

    rotated_p2_rotation_matrix = np.dot(np.transpose(p1_rotation_matrix), p2_rotation_matrix)
    translated_p2_translation_vector = p2_translation_vector - p1_translation_vector
    rotated_p2_translation_vector = np.dot(np.transpose(p1_rotation_matrix), translated_p2_translation_vector)
    rotated_p2_translation_vector += p1_translation_vector

    rotated_p1_rotation_matrix = np.dot(np.transpose(p1_rotation_matrix), p1_rotation_matrix)
    rotated_p1_translation_vector = p1_translation_vector

    rotated_pose1 = np.eye(3, 4)
    rotated_pose1[:, :3] = rotated_p1_rotation_matrix
    rotated_pose1[:, 3] = rotated_p1_translation_vector

    rotated_pose2 = np.eye(3, 4)
    rotated_pose2[:, :3] = rotated_p2_rotation_matrix
    rotated_pose2[:, 3] = rotated_p2_translation_vector

    point_1_2 = transform_points_to_world_coords(rotated_pose1)
    point_3_4 = transform_points_to_world_coords(rotated_pose2)

    # Convert open3D point cloud to numpy array
    pcd_array = np.asarray(point_cloud.points)

    rotated_selection = select_points_in_volume(pcd_array, point_1_2[0], point_1_2[1], point_3_4[0], point_3_4[1])

    selection_pcd = o3d.geometry.PointCloud()
    selection_pcd.points = o3d.utility.Vector3dVector(rotated_selection)

    selection_pcd.rotate(p1_rotation_matrix, center=p1_translation_vector)

    return np.asarray(selection_pcd.points)
"""


def road_cross_section_plot(in_selection, out_selection, gt_selection, point_1234, pose_1):

    #input_selection = o3d.geometry.PointCloud()
    #input_selection.points = o3d.utility.Vector3dVector(in_selection)
    _, _, input_selection = select_points_in_volume(in_selection, np.append(point_1234[7:11], point_1234[11:15], axis=0))

    output_selection = o3d.geometry.PointCloud()
    output_selection.points = o3d.utility.Vector3dVector(out_selection)
    _, _, output_selection = select_points_in_volume(output_selection, np.append(point_1234[7:11], point_1234[11:15], axis=0))

    _, _, gt_selection = select_points_in_volume(gt_selection, np.append(point_1234[7:11], point_1234[11:15], axis=0))

    #o3d.visualization.draw_geometries([output_selection])

    rotation_matrix, translation_vector = decompose_extrinsic_matrix(pose_1)

    input_selection = input_selection.rotate(np.transpose(rotation_matrix), center=translation_vector)
    output_selection = output_selection.rotate(np.transpose(rotation_matrix), center=translation_vector)
    gt_selection = gt_selection.rotate(np.transpose(rotation_matrix), center=translation_vector)


    points = np.asarray(output_selection.points)
    # Compute translation vector to move center of rotation to origin
    center = np.mean(points, axis=0)

    rotation_angle = 0.1

    # Rotate pointcloud so that start and end are on same height
    output_selection = rotate_pcd(output_selection, [0, 0, rotation_angle], center=center)
    input_selection = rotate_pcd(input_selection, [0, 0, rotation_angle], center=center)
    gt_selection = rotate_pcd(gt_selection, [0, 0, rotation_angle], center=center)


    old_pc = np.asarray(input_selection.points)
    new_pc = np.array(output_selection.points)
    gt_pc = np.array(gt_selection.points)

    # Extract Y and Z coordinates from the selected coordinates
    y_values_o = -old_pc[:, 1]
    z_values_o = old_pc[:, 0]

    y_values_n = -new_pc[:, 1]
    z_values_n = new_pc[:, 0]

    y_values_gt = -gt_pc[:, 1]
    z_values_gt = gt_pc[:, 0]

    y_values_o -= np.abs(np.mean(y_values_n))
    z_values_o += np.abs(np.min(z_values_n))
    y_values_gt -= np.abs(np.mean(y_values_n))
    z_values_gt += np.abs(np.min(z_values_n))
    y_values_n -= np.abs(np.mean(y_values_n))
    z_values_n += np.abs(np.min(z_values_n))

    # Set the figure size
    fig, ax = plt.subplots(figsize=(11, 4))  # Adjust the width and height as desired

    ax.set_axisbelow(True)

    # Plot Y against Z
    ax.scatter(z_values_o, y_values_o, color='silver', label='MonoRec')
    #ax.scatter(z_values_gt, y_values_gt, color='grey', label='LiDAR')
    ax.scatter(z_values_n, y_values_n, color='blue', label='post processing')

    # Set labels for the axes
    ax.set_xlabel('road width [meters]', fontsize=16)
    ax.set_ylabel('road height [meters]', fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set title for the plot
    #ax.set_title('road cross section', fontsize=16)

    ax.legend(fontsize=16, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.grid()
    plt.ylim([-0.2, 0.2])
    #plt.xlim([-0.5, 7.8])
    plt.tight_layout()
    plt.savefig('plot_05_2742-2743_plot_2', dpi=300)
    plt.show()


def road_longitudinal_profile_plot(pcd, pcd_gt, pcd_reconstruction, point_1234, pose_1):

    _, _, input_selection = select_points_in_volume(pcd, np.append(point_1234[15:19], point_1234[19:23], axis=0))

    _, _, output_selection = select_points_in_volume(pcd_reconstruction, np.append(point_1234[15:19], point_1234[19:23], axis=0))

    _, _, gt_selection = select_points_in_volume(pcd_gt, np.append(point_1234[15:19], point_1234[19:23], axis=0))

    o3d.visualization.draw_geometries([output_selection])

    rotation_matrix, translation_vector = decompose_extrinsic_matrix(pose_1)

    input_selection = input_selection.rotate(np.transpose(rotation_matrix), center=translation_vector)
    output_selection = output_selection.rotate(np.transpose(rotation_matrix), center=translation_vector)
    gt_selection = gt_selection.rotate(np.transpose(rotation_matrix), center=translation_vector)


    ##### Rotate pointcloud so that start and end are on same height
    # Computation
    points = np.asarray(output_selection.points)
    # Find indices of points with minimum and maximum X values
    min_x_idx = np.argmin(points[:, 2])
    max_x_idx = np.argmax(points[:, 2])
    # Calculate desired Y value based on minimum and maximum X values
    desired_y = (points[min_x_idx, 1] + points[max_x_idx, 1]) / 2.0
    # Compute translation vector to move center of rotation to origin
    center = np.mean(points, axis=0)
    # Compute rotation matrix to align desired Y value with Y axis
    rotation_angle = np.arctan2(desired_y, points[min_x_idx, 1])

    print('Rotation angle from long. plot: ', rotation_angle)

    rotation_angle = -0.97

    # Rotate pointcloud so that start and end are on same height
    output_selection = rotate_pcd(output_selection, [rotation_angle, 0, 0], center=center)
    input_selection = rotate_pcd(input_selection, [rotation_angle, 0, 0], center=center)
    gt_selection = rotate_pcd(gt_selection, [rotation_angle, 0, 0], center=center)

    old_pc = np.asarray(input_selection.points)
    new_pc = np.array(output_selection.points)
    gt_pc = np.array(gt_selection.points)

    # Extract Y and Z coordinates from the selected coordinates
    y_values_o = -old_pc[:, 1]
    z_values_o = old_pc[:, 2]

    y_values_n = -new_pc[:, 1]
    z_values_n = new_pc[:, 2]

    y_values_gt = -gt_pc[:, 1]
    z_values_gt = gt_pc[:, 2]

    y_values_o -= np.abs(np.mean(y_values_n))
    z_values_o -= np.abs(np.min(z_values_n))
    y_values_gt -= np.abs(np.mean(y_values_n))
    z_values_gt -= np.abs(np.min(z_values_n))
    y_values_n -= np.abs(np.mean(y_values_n))
    z_values_n -= np.abs(np.min(z_values_n))

    # Set the figure size
    fig, ax = plt.subplots(figsize=(11, 4))  # Adjust the width and height as desired

    ax.set_axisbelow(True)

    # Plot Y against Z
    ax.scatter(z_values_o, y_values_o, color='silver', label='MonoRec')
    #ax.scatter(z_values_gt, y_values_gt, color='grey', label='LiDAR')
    ax.scatter(z_values_n, y_values_n, color='blue', label='post processing')

    # Set labels for the axes
    ax.set_xlabel('road depth [meters]', fontsize=16)
    ax.set_ylabel('road height [meters]', fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set title for the plot
    #ax.set_title('road cross section', fontsize=16)

    ax.legend(fontsize=16, loc='lower right')

    plt.ylim([-0.1, 0.1])
    #plt.xlim([0, 9])
    plt.tight_layout()
    plt.grid()
    #plt.savefig('plot_long_mid_plot_3.png', dpi=300)
    plt.show()



@gin.configurable
def extract_road(point_cloud, point_cloud_gt, post_processing_mode):

    poses_list, long_plot_start_marker = get_poses()

    extracted_road_points = []

    start_time = time.time()

    for i in range(len(poses_list)-1):
        extrinsic_pose1 = poses_list[i]
        extrinsic_pose2 = poses_list[i+1]

        point_1234 = transform_points_to_world_coords(extrinsic_pose1)
        point_5678 = transform_points_to_world_coords(extrinsic_pose2)

        corner_points = np.append(point_1234[0:4], point_5678[0:4], axis=0)

        if i==0:
            start_pose = point_1234[4]
        if i==len(poses_list)-2:
            end_pose = point_5678[4]
        if i==int((len(poses_list)-1)/2):
            middle_pose = point_1234[4]
            middle_pose[1] = middle_pose[1] + 1.65

        if i == long_plot_start_marker:
            saved_point_1234 = point_1234
            saved_pose_1 = extrinsic_pose1

        selection, oriented_bounding_box, full_height_selection = select_points_in_volume(point_cloud, corner_points)
        input_selection = full_height_selection

        _, _, full_height_selection = select_points_in_volume(point_cloud_gt, corner_points)

        gt_selection = full_height_selection

        if len(selection) == 0:
            continue

        if post_processing_mode == 'alternating':
            selection = slice_in_given_dimension(selection)

            output_selection = selection

            if (len(poses_list)-1) == 1:
                # Plot road cross section, should be performed for one frame only
                road_cross_section_plot(input_selection, output_selection, gt_selection, point_1234, extrinsic_pose1)


        if extracted_road_points == []:
            extracted_road_points = selection
        else:
            extracted_road_points = np.append(extracted_road_points, selection, axis=0)

    if post_processing_mode == 'sequential':
        extracted_road_points = slice_in_given_dimension(extracted_road_points)

    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime', runtime)

    extracted_road_pcd = o3d.geometry.PointCloud()
    extracted_road_pcd.points = o3d.utility.Vector3dVector(extracted_road_points)

    if long_plot_start_marker != -1:
        # Visualize longitudinal road profile plot
        road_longitudinal_profile_plot(point_cloud, point_cloud_gt, extracted_road_pcd, saved_point_1234, saved_pose_1)

    #o3d.visualization.draw_geometries([extracted_road_pcd, oriented_bounding_box])

    '''
    extracted_road_pcd_copy = extracted_road_pcd
    
    for i in range(len(poses_list)-1):
        extrinsic_pose1 = poses_list[i]
        extrinsic_pose2 = poses_list[i+1]

        point_1234 = transform_points_to_world_coords(extrinsic_pose1)
        point_5678 = transform_points_to_world_coords(extrinsic_pose2)

        corner_points = np.append(point_1234, point_5678, axis=0)

        selection_road_color, extracted_road_pcd_copy = paint_pcd_height(extracted_road_pcd_copy, corner_points)

        if i == 0:
            extracted_road_color = selection_road_color
        else:
            extracted_road_color = np.append(extracted_road_color, selection_road_color, axis=0)
            
    extracted_road_pcd.colors = o3d.utility.Vector3dVector(extracted_road_color)
    '''

    return extracted_road_pcd, middle_pose

