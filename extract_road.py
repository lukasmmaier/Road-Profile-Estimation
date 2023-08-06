import gin
from absl import app, flags
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

from estimate_road_profile import slice_in_given_dimension
from utils_and_tools.o3d_tools import rotate_pcd

FLAGS = flags.FLAGS


def check_for_inconsistency(frame, start_frame, end_frame):
    """Outputs a marker, that symbolises whether the frame lies between start and end frame or not.
    Args:
        frame (int): Frame number to check for inconsistency.
        start_frame (int): Start frame defining the sequence start.
        end_frame (int): End frame defining the sequence end.
    Returns:
        marker: Frame position relative to the start frame.
    """
    if (frame >= start_frame) and (frame <= end_frame):
        marker = frame-start_frame
    else:
        marker = -1

    return marker


@gin.configurable
def get_poses(poses_path, sequence, start_frame, end_frame, long_plot_start_frame, cross_plot_start_frame):
    """Gets the extrinsic data between the selected start/end frame and returns it as a list
    Args:
        poses_path (str): gin-configurable - Path referring to the poses.
        sequence (str): gin-configurable - Sequence number corresponding to the MonoRec estimate.
        start_frame (int): gin-configurable - Start frame of the sequence.
        end_frame (int): gin-configurable - End frame of the sequence.
        long_plot_start_frame (int): gin-configurable - Frame where the longitudinal section plot is located.
        cross_plot_start_frame (int): gin-configurable - Frame where the cross section plot is located.
    Returns:
        extrinsic_data_list: Array containing all poses relevant for the evaluation.
        long_plot_start_marker: Marker defining the plot position relative to the start frame.
        cross_plot_start_marker: Marker defining the plot position relative to the start frame.
    """
    path = os.path.dirname(os.getcwd())
    path = os.path.join(path, poses_path)

    # Get poses path
    sequence_poses_path = os.path.join(path, sequence + '.txt')

    # Load sequence poses from .txt file
    poses_data = np.loadtxt(sequence_poses_path, dtype='float32')

    if end_frame == -1:
        end_frame = -1
    else:
        end_frame += 1

    # Select keyframe poses from given start, end
    extrinsic_data_list = poses_data[start_frame:end_frame]

    # Marker for longitudinal plot position
    long_plot_start_marker = -1
    cross_plot_start_marker = -1

    if FLAGS.evaluation:
        long_plot_start_marker = check_for_inconsistency(long_plot_start_frame, start_frame, end_frame)
        cross_plot_start_marker = check_for_inconsistency(cross_plot_start_frame, start_frame, end_frame)

    return extrinsic_data_list, long_plot_start_marker, cross_plot_start_marker

@gin.configurable
def select_points_in_volume(pcd, corner_points, maximum_roadway_height=0.5, inverted_height_orientation=True):
    """Selects points within a volume defined by corner points.
    Args:
        pcd (object): Open3D point cloud data.
        corner_points (numpy.array): Array defining the corner points of the bounding box.
        maximum_roadway_height (float): gin-configurable - Variable defining the maximum height of the roadway.
        inverted_height_orientation (bool): gin-configurable - Variable used, if the height information of the initial
                                            pcd is inverted (as in the standard configuration of MonoRec).
    Returns:
        selected_points: Numpy array containing the extracted roadway points.
        point_cloud_crop: Open3D point cloud data containing the full bounding box without maximum_roadway_height.
    """
    # Construct bounding box
    bounding_box_corner_points = corner_points.astype("float64")
    bounding_box_corner_points = o3d.utility.Vector3dVector(bounding_box_corner_points)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bounding_box_corner_points)

    # Crop pcd using bounding box
    point_cloud_crop = pcd.crop(oriented_bounding_box)

    selected_points = np.asarray(point_cloud_crop.points)

    # Delete points above roadway (e.g. trees, ...)
    # Get min/max height values for selected scene and threshold
    if not len(selected_points) == 0:
        # The standard configuration of MonoRec outputs the 3D model upside down e.g. roadway higher than trees
        # If this is the case, the height is inverted leading to inverted_height_orientation = True
        # If not, inverted_height_orientation = False
        if inverted_height_orientation:
            max_y = max(selected_points[:, 1])
            min_y = max_y - maximum_roadway_height
        else:
            min_y = min(selected_points[:, 1])
            max_y = min_y + maximum_roadway_height

        selected_points = selected_points[
            (selected_points[:, 1] >= min_y) &
            (selected_points[:, 1] <= max_y)
        ]

    return selected_points, point_cloud_crop


def paint_pcd_height(extracted_road_pcd, corner_points):
    """NOT USED and NOT STABLE: Is able to colorize single roadway extraction steps based on their height.
    Args:
        extracted_road_pcd (object): Open3D point cloud data.
        corner_points (numpy.array): Array describing the bounding box.
    Returns:
        colors: Color array.
        pcd_cropped_inv: Open3D point cloud data.
    """
    # Crop point cloud using bounding box
    bounding_box = corner_points.astype("float64")
    points = o3d.utility.Vector3dVector(bounding_box)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    point_cloud_crop = extracted_road_pcd.crop(oriented_bounding_box)

    # Delete already used points
    dists = np.asarray(extracted_road_pcd.compute_point_cloud_distance(point_cloud_crop))
    indices = np.where(dists > 0.00001)[0]
    pcd_cropped_inv = extracted_road_pcd.select_by_index(indices)

    # Colorize the selected section based on the height
    selected_points = np.asarray(point_cloud_crop.points)
    mean_height = np.mean(selected_points[:, 1])
    min_height = np.min(selected_points[:, 1])
    max_height = np.max(selected_points[:, 1])
    normalized_height = (selected_points[:, 1] - min_height) / (max_height - min_height)
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_height)
    colors = colors[:, 0:3]

    return colors, pcd_cropped_inv


def decompose_extrinsic_matrix(extrinsic_matrix):
    """Decompose extrinsic matrix into rotation matrix and translation vector.
    Args:
        extrinsic_matrix (numpy.array): Extrinsic matrix
    Returns:
        rotation_matrix: Rotation matrix
        translation_matrix: Translation vector
    """
    extrinsic_matrix = extrinsic_matrix.reshape(3, 4)
    rotation_matrix = extrinsic_matrix[:, :3]
    translation_vector = extrinsic_matrix[:, 3]

    return rotation_matrix, translation_vector


def transform_points_to_world_coords(points, extrinsic_matrix):
    """Transform points from camera coordinates to world coordinates using extrinsic parameters.
    Args:
        points (numpy.array): Array of points constructed in camera coordinates.
        extrinsic_matrix (numpy.array): Extrinsic matrix
    Returns:
        Transformed points (world coordinates)
    """
    rotation_matrix, translation_vector = decompose_extrinsic_matrix(extrinsic_matrix)

    # Transform points to world coordinates
    transformed_points = np.dot(rotation_matrix, points.T) + translation_vector.reshape(-1, 1)

    return transformed_points.T


@gin.configurable
def construct_bounding_box_corners(extrinsic_matrix, roi_bb_width_r=3, roi_bb_width_l=3, roi_bb_dist=0,
                                   eval_cs_plot_pos=0.45, eval_ls_plot_depth=17, eval_ls_plot_pos=0,
                                   eval_cs_plot_depth=0.1, eval_ls_plot_width=0.1):
    """Construct bounding box corners in camera coordinate system and transform to world coordinates.
    Args:
        extrinsic_matrix (numpy.array): Extrinsic matrix
        roi_bb_width_r (float): gin-configurable - Width of the bounding box to the right side of the vehicle
        roi_bb_width_l (float): gin-configurable - Width of the bounding box to the left side of the vehicle
        roi_bb_dist (float): gin-configurable - Distance in the X-Z-plane between the bounding box and the vehicle
        eval_cs_plot_pos (float): gin-configurable - Position from the selected cross section plot frame
        eval_ls_plot_depth (float): gin-configurable - Depth for the longitudinal section plot
        eval_ls_plot_pos (float): gin-configurable - Position for the selected longitudinal plot frame position
        eval_cs_plot_depth (float): gin-configurable - Depth for the cross section plot
        eval_ls_plot_width (float): gin-configurable - Width for the longitudinal section plot
    Returns:
        points: Numpy array containing all points necessary for evaluation or normal functionality
    """
    if FLAGS.evaluation:
        # Points necessary for evaluation (construction of road cross section and longitudinal plots)
        points = np.array([[roi_bb_width_r, 5, roi_bb_dist],
                           [roi_bb_width_r, -5, roi_bb_dist],
                           [-roi_bb_width_l, 5, roi_bb_dist],
                           [-roi_bb_width_l, -5, roi_bb_dist],
                           # Evaluation points for...
                           # ... later rotation
                           [0, 0, 0],
                           [2, 0, 0],
                           [-2, 0, 0],
                           # ... road cross section plot
                           [roi_bb_width_r, 5, eval_cs_plot_pos],
                           [roi_bb_width_r, -5, eval_cs_plot_pos],
                           [-roi_bb_width_l, 5, eval_cs_plot_pos],
                           [-roi_bb_width_l, -5, eval_cs_plot_pos],
                           [roi_bb_width_r, 5, eval_cs_plot_pos + eval_cs_plot_depth],
                           [roi_bb_width_r, -5, eval_cs_plot_pos + eval_cs_plot_depth],
                           [-roi_bb_width_l, 5, eval_cs_plot_pos + eval_cs_plot_depth],
                           [-roi_bb_width_l, -5, eval_cs_plot_pos + eval_cs_plot_depth],
                           # ... road longitudinal section plot
                           [eval_ls_plot_pos, 5, 0.3],
                           [eval_ls_plot_pos, -5, 0.3],
                           [eval_ls_plot_pos + eval_ls_plot_width, 5, 0.3],
                           [eval_ls_plot_pos + eval_ls_plot_width, -5, 0.3],
                           [eval_ls_plot_pos, 5, eval_ls_plot_depth],
                           [eval_ls_plot_pos, -5, eval_ls_plot_depth],
                           [eval_ls_plot_pos + eval_ls_plot_width, 5, eval_ls_plot_depth],
                           [eval_ls_plot_pos + eval_ls_plot_width, -5, eval_ls_plot_depth]
                           ])

    else:
        # Point necessary for normal functionality
        points = np.array([[roi_bb_width_r, 5, roi_bb_dist],
                           [roi_bb_width_r, -5, roi_bb_dist],
                           [-roi_bb_width_l, 5, roi_bb_dist],
                           [-roi_bb_width_l, -5, roi_bb_dist]])

    points = transform_points_to_world_coords(points, extrinsic_matrix)

    return points


def get_and_rotate_pcd_for_plot(pcd, point_1234, rotation_matrix, translation_vector, plot_type,
                                vis_selection=False):
    """Get data and preprocess pcd for plots - part 1.
    Args:
        pcd (object): Open3D point cloud data.
        point_1234 (numpy.array): Array containing the point defining the bounding boxes for the plots.
        rotation_matrix (numpy.array): Camera extrinsic rotation matrix
        translation_vector (numpy.array): Camera extrinsic translation vector
        plot_type (str): Variable defining which plot to do - road longitudinal or road cross section
        vis_selection (bool): Visualize the points used for the plots as a 3D object
    Returns:
        selection: Rotated point selection used for the plot
    """
    if plot_type == 'long':
        p1 = point_1234[15:19]
        p2 = point_1234[19:23]
    elif plot_type == 'cross':
        p1 = point_1234[7:11]
        p2 = point_1234[11:15]
    else:
        print('Unknown plot type! Automatically switching to cross section plot!')
        p1 = point_1234[7:11]
        p2 = point_1234[11:15]

    # Select data using bounding box constructed earlier
    _, selection = select_points_in_volume(pcd, np.append(p1, p2, axis=0))

    if vis_selection:
        o3d.visualization.draw_geometries([selection])

    # Rotate selection, so that the world coordinate axis correspond to camera coordinate axis
    selection = selection.rotate(np.transpose(rotation_matrix), center=translation_vector)

    return selection


def rotate_and_preprocess_pcd_for_plot(pcd, rotation_angle, center, plot_type, inverted_height_orientation=True):
    """Preprocess pcd for plots - part 2.
    Args:
        pcd (object): Open3D point cloud data.
        rotation_angle (float): Angle used to rotate the plot points for visualization purposes.
        center (tuple): Rotation center.
        plot_type (str): Variable defining which plot to do - road longitudinal or road cross section.
        inverted_height_orientation (bool): Variable used, if the height information of the initial pcd is inverted
                                            (as in the standard configuration of MonoRec).
    Returns:
        y_coordinates: Extracted y coordinates form the point selection
        x_or_z_coordinates: Corresponding x or z coordinate from the point selection
    """
    rotation_angle_x = 0
    rotation_angle_z = 0

    if plot_type == 'long':
        rotation_angle_x = rotation_angle
        axis = 2
    elif plot_type == 'cross':
        rotation_angle_z = rotation_angle
        axis = 0
    else:
        print('Unknown plot type! Automatically switching to cross section plot!')
        rotation_angle_z = rotation_angle
        axis = 0

    # Rotate selection
    rotated_pcd = rotate_pcd(pcd, [rotation_angle_x, 0, rotation_angle_z], center=center)

    point_array = np.asarray(rotated_pcd.points)

    if inverted_height_orientation:
        sign = -1
    else:
        sign = 1

    # Extract Y and X/Z coordinates from the selected coordinates
    y_coordinates = sign * point_array[:, 1]
    x_or_z_coordinates = point_array[:, axis]

    return y_coordinates, x_or_z_coordinates


def shift_coordinates(y, x_or_z, mean_y, min_x_or_z):
    """Shift input coordinates using mean and min values - for visualization purposes.
    Args:
        y (numpy.array): Y coordinate.
        x_or_z (numpy.array): Corresponding X or Z coordinates.
        mean_y (float): Mean of the Y coordinates.
        min_x_or_z (float): Minimum value of the X or Z coordinates.
    Returns:
        y: Mean shifted Y coordinate.
        x_or_z: Minimum shifted X or Z coordinate.
    """
    y -= mean_y
    x_or_z -= min_x_or_z

    return y, x_or_z


def preprocess_pcd_for_plot(pcd, pcd_post_processed, point_1234, pose_1, rotation_angle, plot_type, pcd_gt=False):
    """Preprocess pcd for plots.
    Args:
        pcd (object): Open3D point cloud data (=MonoRec output).
        pcd_post_processed (object): Open3D point cloud data (=Post processing output).
        point_1234 (numpy.array): Array containing the point defining the bounding boxes for the plots.
        pose_1 (numpy.array): Extrinsic camera parameters used to improve the plot visualization.
        rotation_angle (numpy.array): Angle used to rotate the plot points for visualization purposes.
        plot_type (str): Variable defining which plot to do - road longitudinal or road cross section.
        pcd_gt (bool): Placeholder - whether to use pcd ground truth.
    Returns:
        input_y_coordinate: MonoRec output (=post processing input) - Y coordinates
        input_x_or_z_coordinate: MonoRec output (=post processing input) - X or Z coordinates
        output_y_coordinate: Post processing output - Y coordinates
        output_x_or_z_coordinate: Post processing output - X or Z coordinates
    """
    rotation_matrix, translation_vector = decompose_extrinsic_matrix(pose_1)

    # Get selection and rotate, so that the world coordinate axis correspond to camera coordinate axis
    input_selection = get_and_rotate_pcd_for_plot(pcd, point_1234, rotation_matrix, translation_vector,
                                                  plot_type)
    output_selection = get_and_rotate_pcd_for_plot(pcd_post_processed, point_1234, rotation_matrix,
                                                   translation_vector, plot_type)
    #gt_selection = ....

    # Rotate selection if rotation angle is given from gin-config
    # Calculate center for rotation
    points = np.asarray(output_selection.points)

    if len(points) == 0:
        return [], [], [], []

    center = np.mean(points, axis=0)

    input_y_coordinates, input_x_or_z_coordinates = rotate_and_preprocess_pcd_for_plot(input_selection, rotation_angle,
                                                                                     center, plot_type)
    output_y_coordinates, output_x_or_z_coordinates = rotate_and_preprocess_pcd_for_plot(output_selection, rotation_angle,
                                                                                       center, plot_type)
    #gt_y_coordinate, gt_x_or_z_coordinate = ...

    # Normalize y coordinate (mean = 0) and shift x or z coordinate (min = 0)
    mean_y = np.mean(output_y_coordinates)
    min_x_or_z = np.min(output_x_or_z_coordinates)

    input_y_coordinate, input_x_or_z_coordinate = shift_coordinates(input_y_coordinates, input_x_or_z_coordinates,
                                                                    mean_y, min_x_or_z)
    #gt_y_coordinate, gt_x_or_z_coordinate = ...
    output_y_coordinate, output_x_or_z_coordinate = shift_coordinates(output_y_coordinates, output_x_or_z_coordinates,
                                                                      mean_y, min_x_or_z)

    return input_y_coordinate, input_x_or_z_coordinate, output_y_coordinate, output_x_or_z_coordinate


@gin.configurable
def road_cross_section_plot(pcd, pcd_post_processed, point_1234, pose_1, rotation_angle, y_axis_lim, pcd_gt=False):
    """Road cross section plot.
    Args:
        pcd (object): Open3D point cloud data (=MonoRec output)
        pcd_post_processed (object): Open3D point cloud data (=Post processing output)
        point_1234 (numpy.array): Array containing the point defining the bounding boxes for the plots.
        pose_1 (numpy.array): Extrinsic camera parameters used to improve the plot visualization.
        rotation_angle (list): gin-configurable - Angle used to rotate the plot points for visualization purposes.
        y_axis_lim (list): gin-configurable - Plot y-axis limit
        pcd_gt (object): Placeholder - whether to use pcd ground truth.
    """
    in_y_coord, in_x_coord, out_y_coord, out_x_coord = preprocess_pcd_for_plot(pcd, pcd_post_processed, point_1234,
                                                                               pose_1, rotation_angle, plot_type='cross')

    if len(out_y_coord) == 0:
        print('No points available for road cross section plot!')
        print('Plot is skipped!')
        return

    # Configure figure
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_axisbelow(True)

    # Plot Y against X
    ax.scatter(in_x_coord, in_y_coord, color='silver', label='MonoRec')
    #ax.scatter(z_values_gt, y_values_gt, color='grey', label='LiDAR')
    ax.scatter(out_x_coord, out_y_coord, color='blue', label='post processing')

    # Configure axes
    ax.set_xlabel('road width [meters]', fontsize=16)
    ax.set_ylabel('road height [meters]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set title, legend, ...
    ax.set_title('road cross section plot', fontsize=16)
    ax.legend(fontsize=16, loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid()
    plt.ylim(y_axis_lim)
    plt.tight_layout()
    #plt.savefig('plot_cross.png', dpi=300)
    plt.show()

@gin.configurable
def road_longitudinal_profile_plot(pcd, pcd_post_processed, point_1234, pose_1, rotation_angle, y_axis_lim, pcd_gt=False):
    """Road longitudinal section plot => Y-Z-Plot (Height-Depth-Plot).
        Args:
            pcd (object): Open3D point cloud data (=MonoRec output)
            pcd_post_processed (object): Open3D point cloud data (=Post processing output)
            point_1234 (numpy.array): Array containing the point defining the bounding boxes for the plots.
            pose_1 (numpy.array): Extrinsic camera parameters used to improve the plot visualization.
            rotation_angle (list): gin-configurable - Angle used to rotate the plot points for visualization purposes.
            y_axis_lim (list): gin-configurable - Plot y-axis limit
            pcd_gt (object): Placeholder - whether to use pcd ground truth.
    """
    in_y_coord, in_z_coord, out_y_coord, out_z_coord = preprocess_pcd_for_plot(pcd, pcd_post_processed, point_1234,
                                                                               pose_1, rotation_angle, plot_type='long')

    if len(out_y_coord) == 0:
        print('No points available for longitudinal cross section plot!')
        print('Plot is skipped!')
        return

    # Configure figure
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_axisbelow(True)

    # Plot Y against Z
    ax.scatter(in_z_coord, in_y_coord, color='silver', label='MonoRec')
    #ax.scatter(z_values_gt, y_values_gt, color='grey', label='LiDAR')
    ax.scatter(out_z_coord, out_y_coord, color='blue', label='post processing')

    # Configure axes
    ax.set_xlabel('road depth [meters]', fontsize=16)
    ax.set_ylabel('road height [meters]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set title, legend, ...
    ax.set_title('road longitudinal section plot', fontsize=16)
    ax.legend(fontsize=16, loc='lower right')
    plt.ylim(y_axis_lim)
    plt.tight_layout()
    plt.grid()
    #plt.savefig('plot_long.png', dpi=300)
    plt.show()


@gin.configurable
def extract_road(pcd, post_processing_mode, pcd_gt=False):
    """Roadway extraction step.
    Args:
        pcd (object): Open3D point cloud data.
        post_processing_mode (str): gin-configurable - Mode defining if the road extraction and the denoising should be
                                    performed alternating for each frame or sequential after the whole section.
                                    First mode simulates the real behaviour, second mode improves the visualization for
                                    a qualitative evaluation.
        pcd_gt (object): Placeholder - whether to use pcd ground truth.
    Returns:
        pcd_post_processed: Post processed Open3D point cloud data.
        center: Center of pcd_post_processed used for later rotation.
    """
    # In regular mode, set post processing mode to alternating
    if not FLAGS.evaluation:
        post_processing_mode = 'alternating'

    # Get poses for extraction
    poses_list, long_plot_start_marker, cross_plot_start_marker = get_poses()
    point_array = []
    center = False

    # Runtime measurement
    start_time = time.time()

    # Loop for roadway extraction; Iterating over the poses
    for i in range(len(poses_list)-1):

        # Camera pose timestep t-1; Past vehicle position
        extrinsic_pose1 = poses_list[i]
        # Camera pose timestep t; Current vehicle position
        extrinsic_pose2 = poses_list[i+1]

        # Get all points necessary for bounding box operation
        point_1234 = construct_bounding_box_corners(extrinsic_pose1)
        point_5678 = construct_bounding_box_corners(extrinsic_pose2)

        # Get points necessary for bounding box construction => corner points of bounding box
        corner_points = np.append(point_1234[0:4], point_5678[0:4], axis=0)

        # Roadway extraction step
        raw_selection, full_height_raw_selection = select_points_in_volume(pcd, corner_points)

        # Save necessary points for evaluation/plots
        if FLAGS.evaluation:
            # Do roadway extraction also for ground truth point cloud
            #_, full_height_raw_selection_gt = select_points_in_volume(pcd_gt, corner_points)

            # Save a point in the middle of the extracted roadway (center) for later rotation
            if i == int((len(poses_list)-1)/2):
                center = point_1234[4]
                center[1] = center[1] + 1.65

            # Save values for road longitudinal section plot
            if i == long_plot_start_marker:
                long_plot_point_1234 = point_1234
                long_plot_pose_1 = extrinsic_pose1

            # Save values for road cross section plot
            if i == cross_plot_start_marker:
                cross_plot_point_1234 = point_1234
                cross_plot_pose_1 = extrinsic_pose1

        # Skip roadway sections where no points could be extracted
        if len(raw_selection) == 0:
            continue

        if post_processing_mode == 'alternating':
            point_selection = slice_in_given_dimension(raw_selection)
        else:
            point_selection = raw_selection

        if point_array == []:
            point_array = point_selection
        else:
            point_array = np.append(point_array, point_selection, axis=0)

    # Calculate and print runtime
    end_time = time.time()
    runtime = end_time - start_time
    print('Runtime:', runtime)

    if post_processing_mode == 'sequential':
        point_array = slice_in_given_dimension(point_array)

    # Convert numpy array to pcd
    pcd_post_processed = o3d.geometry.PointCloud()
    pcd_post_processed.points = o3d.utility.Vector3dVector(point_array)

    if FLAGS.evaluation:
        if long_plot_start_marker != -1:
            # Visualize road longitudinal section plot
            road_longitudinal_profile_plot(pcd, pcd_post_processed, long_plot_point_1234, long_plot_pose_1)
        if cross_plot_start_marker != -1:
            # Visualize road cross section plot
            road_cross_section_plot(pcd, pcd_post_processed, cross_plot_point_1234, cross_plot_pose_1)

    return pcd_post_processed, center
