import gin
import math
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt


def bounding_box(points, dim, min_bound, max_bound):
    """ Taken from:  https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy
    Compute a bounding_box filter on the given points
    """
    bb_filter = np.logical_and(points[:, dim] > min_bound, points[:, dim] < max_bound)

    return bb_filter


def histogram_from_bounding_boxes(histogram_points):
    ''' Plot histogram of box plots from point collection. '''
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


def initialize_for_loops(pcd_array, slice_dimension, hist_window_size, bb_free):
    # Calculate the minimum possible value in given dimension
    min_value = np.min(pcd_array[:, slice_dimension])

    # Calculate number of iterations
    # Get the maximum possible value in given dimension
    max_value = np.max(pcd_array[:, slice_dimension])
    # Calculate how many bounding boxes fit between max_value and min_value
    num_interations = math.ceil(max_value - min_value) / (hist_window_size * bb_free)

    return min_value, max_value, num_interations


@gin.configurable
def reconstruct_mesh(road_topology, mesh_method):
    # radii = [0.005, 0.01, 0.02, 0.04]
    radii = [0.01, 0.02, 0.04, 0.08]
    # radii = [0.1, 0.2, 0.4, 0.8]
    radius = 0.1
    max_nn = 30

    do_visualization = True

    # Estimate vertex normals
    road_topology.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    # Reverse direction (for visualization purposes
    normals = np.asarray(road_topology.normals)
    road_topology.normals = o3d.utility.Vector3dVector(-normals)

    if mesh_method == 'ball_pivoting':
        # BALL PIVOTING method
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(road_topology,
                                                                               o3d.utility.DoubleVector(radii))

    elif mesh_method == 'poisson':
        # POISSON method
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(road_topology, depth=9)

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
        o3d.visualization.draw_geometries([road_topology, mesh])

    #o3d.io.write_triangle_mesh("poisson_rec.obj", mesh)


@gin.configurable
def slice_in_given_dimension(pcd, hist_window_size, bb_overlap, slice_dimensions, estimation_variant, plot_histogram,
                             plot_histogram_first_dim_position, surface_mesh):
    # Convert open3D point cloud to numpy array
    pcd_array = np.asarray(pcd.points)

    # Initialize lists
    est_road_topology, histogram_points = [], []

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
    first_dim_bound, _, num_iterations_first_dim = initialize_for_loops(pcd_array, first_dim,
                                                                        hist_window_size, bb_free)

    # Calculate the position from where the histogram should be calculated
    histogram_position = int(num_iterations_first_dim * plot_histogram_first_dim_position)

    # FIRST: Slice in first dimension
    for bb_first_dim in range(int(num_iterations_first_dim)):
        # Define bounds of bounding box
        min_first_dim_bound = first_dim_bound
        max_first_dim_bound = first_dim_bound + hist_window_size

        # Slice point cloud in given dimension using bounding box
        bb_filter = bounding_box(pcd_array, first_dim, min_first_dim_bound, max_first_dim_bound)

        # Get points inside bounding box from point cloud
        slice_pcd_array = pcd_array[bb_filter]

        # Move window
        first_dim_bound = first_dim_bound + hist_window_size*bb_free

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
            if plot_histogram:
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
                new_y_coordinate = np.mean(bb_pcd_array[:, 1])
            elif estimation_variant == 'median':
                new_y_coordinate = np.median(bb_pcd_array[:, 1])
            else:
                print('No method selected to calculate y-Coordinate: proceed with median')
                new_y_coordinate = np.median(bb_pcd_array[:, 1])

            # Initialize
            estimate_road_coordinate = [0, new_y_coordinate, 0]

            estimate_road_coordinate[first_dim] = min_first_dim_bound + hist_window_size/2
            estimate_road_coordinate[second_dim] = min_second_dim_bound + hist_window_size/2

            # Append new coordinate points to list
            est_road_topology.append(estimate_road_coordinate)

    # Show point cloud with road topology estimation
    road_topology = o3d.geometry.PointCloud()
    road_topology.points = o3d.utility.Vector3dVector(est_road_topology)

    # Plot histogram of road crossing
    if plot_histogram:
        histogram_from_bounding_boxes(histogram_points)

    # Weather to subsequently reconstruct surface mesh or only visualize point cloud
    if surface_mesh:
        # Reconstruct mesh from point cloud
        reconstruct_mesh(road_topology)
    else:
        # Visualize road topology point cloud
        o3d.visualization.draw_geometries([road_topology])
