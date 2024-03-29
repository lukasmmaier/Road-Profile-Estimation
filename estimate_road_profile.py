import gin
from absl import app, flags
import math
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

from utils_and_tools.o3d_tools import estimate_vertex_normals

FLAGS = flags.FLAGS


def bounding_box(points, dim, min_bound, max_bound):
    """Compute a bounding_box filter based the given bounds.
    Args:
        points (numpy array): Points of the point cloud data.
        dim (list): Defines in which dimension the bounding box should be constructed.
        min_bound (float): Minimum bound used for bounding box selection.
        max_bound (float): Maximum bound used for bounding box selection.
    Returns:
        bb_filter: Numpy array defining whether a point lies inside the bounding box or not.
    """
    bb_filter = np.logical_and(points[:, dim] > min_bound, points[:, dim] < max_bound)

    return bb_filter


def histogram_from_bounding_boxes(histogram_points):
    """Plot histogram of box plots from point collection.
    Args:
        histogram_points (list): List containing lists containing bounding box points.
    """
    # Calling DataFrame constructor on list
    df_est_road_topology = pd.DataFrame(histogram_points, columns=['X', 'Y', 'Z', 'bb_x'])

    # Plot histogram
    df_est_road_topology.boxplot(column='Y', by='bb_x', grid=False, figsize=(18, 4))

    # Change histogram setting
    plt.ylim([0.35, 0.75])
    plt.suptitle("")
    plt.title("Road cross section box plot")
    plt.xlabel("Bounding boxes over X-axis")
    plt.ylabel("Y-axis (height)")

    # Show histogram
    plt.show()


def initialize_for_loops(pcd, slice_dimension, hist_window_size, bb_free):
    """Initialize for loops for denoising.
    Args:
        pcd (object): Open3D point cloud data.
        slice_dimension (int): Dimension in which the bounding boxes should be constructed.
        hist_window_size (float): Quadratic size (Width and depth) of the bounding box.
        bb_free (float): NOT overlapping amount (in percent) of the bounding boxes.
    Returns:
        min_value: Minimum value in the selected dimension
        max_value: Maximum value in the selected dimension
        num_iterations: Number of bounding boxes between the maximum and the minimum values
    """
    # Calculate the minimum possible value in given dimension
    min_value = np.min(pcd[:, slice_dimension])

    # Calculate number of iterations
    # Get the maximum possible value in given dimension
    max_value = np.max(pcd[:, slice_dimension])
    # Calculate how many bounding boxes fit between max_value and min_value
    num_interations = math.ceil(max_value - min_value) / (hist_window_size * bb_free)

    return min_value, max_value, num_interations


def reconstruct_mesh(est_road_profile, mesh_method, radii, radius, max_nn, do_visualization):
    """NOT USED and NOT STABLE: Reconstruct mesh from point cloud.
    Args:
        est_road_profile (object): Open3D point cloud data (=post processing output)
        mesh_method (str): Methode used for mesh reconstruction.
        radii (list): Variable used for ball pivoting mesh reconstruction algorithm.
        radius (float): Variable used for vertex normal estimation.
        max_nn (float): Variable used for vertex normal estimation.
        do_visualization (bool): Variable deciding if the mesh should be visualized after reconstruction.
    """
    # Estimate vertex normals
    est_road_profile = estimate_vertex_normals(est_road_profile, radius, max_nn)

    # Reverse direction (for visualization purposes
    normals = np.asarray(est_road_profile.normals)
    est_road_profile.normals = o3d.utility.Vector3dVector(-normals)

    if mesh_method == 'ball_pivoting':
        # BALL PIVOTING method
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(est_road_profile,
                                                                               o3d.utility.DoubleVector(radii))

    elif mesh_method == 'poisson':
        # POISSON method
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(est_road_profile, depth=9)

        # Estimate vertex normals (for some reason they get lost in the Poisson method)
        mesh.compute_vertex_normals()

        # Paint mesh grey
        mesh.paint_uniform_color([0.8, 0.8, 0.8])

        # Remove all mesh which were constructed from low amount from points (remove wrong estimations)
        vertices_to_remove = densities < np.quantile(densities, 0.2)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    else:
        print("No suitable mesh reconstruction method chosen! Did not perform any reconstruction")
        do_visualization = False

    if do_visualization:
        # Visualize reconstructed mesh + road topology point cloud
        o3d.visualization.draw_geometries([est_road_profile, mesh])


@gin.configurable
def slice_in_given_dimension(pcd, hist_window_size, bb_overlap, slice_dimensions, estimation_variant, plot_histogram,
                             plot_histogram_first_dim_position):
    """Denoising module.
    Args:
        pcd (object): Open3D point cloud data (=Roadway extraction module output)
        hist_window_size (float): gin-configurable - Quadratic size (Width and depth) of the bounding box.
        bb_overlap (float): gin-configurable - Overlapping amount (in percent) of the bounding boxes.
        slice_dimensions (list): gin-configurable - Consecutive dimensions in which the bounding boxes should be constructed.
        estimation_variant (str): gin-configurable - Denoising variant used to denoise the single bounding boxes.
        plot_histogram (bool): gin-configurable - Whether to plot a histogram of the bounding boxes.
        plot_histogram_first_dim_position (int): gin-configurable - Defines the position of the histogram plot
    Returns:
        est_road_profile: Denoised Open3D point cloud data (=post processing output)
    """
    # Initialize lists
    est_road_profile, histogram_points = [], []

    # Get dimensions to slice from list
    first_dim = slice_dimensions[0]
    second_dim = slice_dimensions[1]

    # Convert bounding box overlap to how many percentage of the bounding box is NOT overlapped
    bb_free = 1 - bb_overlap

    # Prevent impossible values
    if bb_free <= 0.01:
        bb_free = 0.01
        print('Bounding box overlap was chosen too big!')
        print('Continue with bb_overlap = 0.99!')
    elif bb_free > 1:
        bb_free = 1
        print('Bounding box overlap was chosen negative! Computation not possible!')
        print('Continue with bb_overlap = 0!')

    # Calculate how many bounding boxes fit in between
    first_dim_bound, _, num_iterations_first_dim = initialize_for_loops(pcd, first_dim,
                                                                        hist_window_size, bb_free)

    # Calculate the position from where the histogram should be calculated
    histogram_position = int(num_iterations_first_dim * plot_histogram_first_dim_position)

    # FIRST: Slice in first dimension
    for bb_first_dim in range(int(num_iterations_first_dim)):
        # Define bounds of bounding box
        min_first_dim_bound = first_dim_bound
        max_first_dim_bound = first_dim_bound + hist_window_size

        # Slice point cloud in given dimension using bounding box
        bb_filter = bounding_box(pcd, first_dim, min_first_dim_bound, max_first_dim_bound)

        # Get points inside bounding box from point cloud
        slice_pcd_array = pcd[bb_filter]

        # Move window
        first_dim_bound = first_dim_bound + hist_window_size*bb_free

        # Continue loop if bounding box is empty
        if len(slice_pcd_array) == 0:
            continue

        # Calculate how many bounding boxes fit in between
        second_dim_bound, _, num_iterations_second_dim = initialize_for_loops(slice_pcd_array, second_dim,
                                                                              hist_window_size, bb_free)

        # SECOND: Slice in second dimension
        for bb_second_dim in range(int(num_iterations_second_dim)):
            # Define bounds of bounding box
            min_second_dim_bound = second_dim_bound
            max_second_dim_bound = second_dim_bound + hist_window_size

            # Slice point cloud in given dimension using bounding box
            bb_filter = bounding_box(slice_pcd_array, second_dim, min_second_dim_bound, max_second_dim_bound)

            # Get points inside bounding box from point cloud
            bb_pcd_array = slice_pcd_array[bb_filter]

            # Save points for histogram visualization
            if FLAGS.evaluation and plot_histogram:
                if bb_first_dim == histogram_position:
                    # Save points for histogram visualization and mark which bounding box belongs to it
                    for point in bb_pcd_array.tolist():
                        # Append bounding box number to point
                        point.append(bb_second_dim)
                        histogram_points.append(point)

            # Move window
            second_dim_bound = second_dim_bound + hist_window_size * bb_free

            # Continue loop if bounding box is empty
            if len(bb_pcd_array) == 0:
                continue

            # Choose which variant is used as estimate for the y-coordinate
            if estimation_variant == 'mean':
                estimate_road_coordinate = np.mean(bb_pcd_array, axis=0)
            elif estimation_variant == 'median':
                estimate_road_coordinate = np.median(bb_pcd_array, axis=0)
            else:
                print('No method selected to calculate y-Coordinate: proceed with median')
                estimate_road_coordinate = np.median(bb_pcd_array, axis=0)

            # Append new coordinate points to list
            est_road_profile.append(estimate_road_coordinate)

    # Plot histogram of road crossing
    if FLAGS.evaluation and plot_histogram:
        histogram_from_bounding_boxes(histogram_points)

    return est_road_profile
