import os
import sys
import open3d as o3d
import numpy as np
import pandas as pd
import itertools
import math
import matplotlib.pyplot as plt

rotate_pc = True
show_pc = True
downsample_pc = False
paint_pc_uniform = False
crop_pc = True
remove_outliers = False
save_pc_to_ply = False
surface_reconstruction_type = False #possible values: 'alpha', 'ball', 'poisson'
histogram_pc = True
estimation_variant = 'median'

# Configure paths
path = '/home/lukas/PycharmProjects/MonoRec/saved/pointclouds/monorec'
#filename = 'seequence_00_6fr.ply'
#filename = 'seequence_00_4fr.ply'
filename = 'seequence_00_2fr.ply'

# ===================== Configuration ==================================================================================
# Downsample
voxel_size = 0.1
# Estimate vertex normal
radius = 0.1
max_nn = 30
# Paint point cloud
#color = [1, 0.706, 0]
color = [0.8, 0.8, 0.8]
# Crop
#bounds = [[-31.8, -24], [-math.inf, 0.75], [-math.inf, math.inf]]
bounds = [[-31.8, -24], [-math.inf, 0.75], [-260, -240]]
#bounds = [[-31.8, -24], [-math.inf, 0.75], [-260, -258]]
# Surface reconstruction
# Alpha shapes
alpha=0.5
# Ball pivoting
# radii = [0.005, 0.01, 0.02, 0.04]
radii = [0.01, 0.02, 0.04, 0.08]
# Poisson surface reconstruction

# Histogram
z_bound = -259
#z_bound = -250.2
hist_window_size = 0.1

# ===================== Functions + load data ==========================================================================
# Visualization function for outlier removal
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# Get filepath from configuration
filepath = os.path.join(path, filename)

# Load point cloud
pcd = o3d.io.read_point_cloud(filepath)


# ===================== Post-Process PC ================================================================================
# Rotate point cloud
if rotate_pc:
    #R = pcd.get_rotation_matrix_from_xyz((np.pi, -np.pi/2, 0))
    R = pcd.get_rotation_matrix_from_xyz(((177.4 / 180) * np.pi, (-92.5 / 180) * np.pi, 0))
    pcd = pcd.rotate(R, center=(0, 0, 0))


points = [
    [-32, 0.8, z_bound-hist_window_size],
    [-32, 0.8, z_bound+hist_window_size],
    [-32, 0.3, z_bound-hist_window_size],
    [-32, 0.3, z_bound+hist_window_size],
    [-24, 0.8, z_bound-hist_window_size],
    [-24, 0.8, z_bound+hist_window_size],
    [-24, 0.3, z_bound-hist_window_size],
    [-24, 0.3, z_bound+hist_window_size],
]
lines = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [0, 6],
    [1, 7],
    [2, 4],
    [3, 5],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set, pcd], mesh_show_wireframe=True)
#o3d.visualization.draw_geometries([pcd])


# Crop point cloud
if crop_pc:
    # Create bounding box X, Y, Z:
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

    # Crop the point cloud using the bounding box:
    pcd = pcd.crop(bounding_box)

# Remove outliers
if remove_outliers:
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)
    display_inlier_outlier(pcd, ind)

# Downsample point cloud and show it
if downsample_pc:
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# Estimate vertex normal
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
#pcd.orient_normals_consistent_tangent_plane(100)

# Paint point cloud in uniform color
if paint_pc_uniform:
    pcd.paint_uniform_color(color)


# ===================== Surface reconstruction =========================================================================
# Alpha shapes [Edelsbrunner1983]
if surface_reconstruction_type == 'alpha':
    pcd = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
    #mesh.compute_vertex_normals()

# Ball pivoting [Bernardini1999]
if surface_reconstruction_type == 'ball':
    pcd = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    
# Poisson surface reconstruction [Kazhdan2006]
if surface_reconstruction_type == 'poisson':
    pass # placeholder


# ===================== Show/Save data =================================================================================
# Show point cloud
if show_pc:
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set, pcd], mesh_show_wireframe=True)
    #o3d.visualization.draw_geometries([pcd])

# Save point cloud to ply
if save_pc_to_ply:
    o3d.io.write_point_cloud("pc_00_06fr.ply", pcd)


# ===================== Calcuate Histogram of Point cloud ==============================================================
def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf, max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Taken from:  https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy
    Compute a bounding_box filter on the given points

    Parameters
    ----------
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


if histogram_pc:
    # Convert open3D point cloud to numpy array
    pc_array = np.asarray(pcd.points)

    # Z-Axis: Define bounds of sliding window
    min_z = z_bound - hist_window_size/2
    max_z = z_bound + hist_window_size/2

    # Initialize
    x_bound = np.min(pc_array[:, 0])

    # Calculate number of iterations
    max_value = np.max(pc_array[:, 0])
    num_iterations = abs((abs(math.ceil(max_value)) - abs(math.ceil(x_bound)))) / hist_window_size

    # FIRST: Cut in z-direction - Get points from sliding window (bounding box)
    bb_filter = bounding_box(pc_array,
                             min_x=-np.inf, max_x=np.inf,
                             min_y=-np.inf, max_y=np.inf,
                             min_z=min_z, max_z=max_z)
    z_filtered_pc_array = pc_array[bb_filter]

    # Initialize lists
    est_road_topology = []
    lines = []
    histogram_points = []

    # SECOND: Cut in x-direction - Get points from sliding window (bounding box)
    for bb_x in range(int(num_iterations)):
        # X-Axis: Define bounds of sliding window
        min_x = x_bound
        max_x = x_bound + hist_window_size

        # Get points in sliding window (bounding box)
        bb_filter = bounding_box(z_filtered_pc_array,
                                 min_x=min_x, max_x=max_x,
                                 min_y=-np.inf, max_y=np.inf,
                                 min_z=-np.inf, max_z=np.inf)

        # Save points for histogram visualization and mark which bounding box belongs to it
        for point in z_filtered_pc_array[bb_filter].tolist():
            # Append bounding box number to point
            point.append(bb_x)
            histogram_points.append(point)

        # Move window
        x_bound = x_bound + hist_window_size

        # Calculate mean, standard deviation and median from bounding box
        bb_mean = np.mean(z_filtered_pc_array[bb_filter][:, 1])
        bb_std = np.std(z_filtered_pc_array[bb_filter][:, 1])
        bb_median = np.median(z_filtered_pc_array[bb_filter][:, 1])

        # Calculate new estimated 2.5D coordinate point
        new_x_coordinate = x_bound + hist_window_size/2
        new_z_coordinate = z_bound

        if estimation_variant == 'mean':
            new_y_coordinate = bb_mean
        elif estimation_variant == 'median':
            new_y_coordinate = bb_median
        else:
            print('No method selected to calculate y-Corrdinate: proceed with median')
            new_y_coordinate = bb_median

        # Set 2.5D coordinate point
        bb_median_coordinates = np.array([new_x_coordinate, new_y_coordinate, z_bound])

        # Append new coordinate points to list
        est_road_topology.append(bb_median_coordinates)

        # Generate line connection between coordinates for visualization
        if bb_x != 0:
            lines.append([bb_x - 1, bb_x])

    # Convert both lists to numpy array for open3D visualization
    points = np.array(est_road_topology)
    lines = np.array(lines)

    # Generate line objects for open3D visualization
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Calling DataFrame constructor on list
    df_est_road_topology = pd.DataFrame(histogram_points, columns=['X', 'Y', 'Z', 'bb_x'])

    # Show histogram
    df_est_road_topology.boxplot(column='Y', by='bb_x', grid=False, figsize=(18, 4))
    plt.ylim([0.35, 0.75])
    plt.suptitle("")
    plt.title("Road cross section box plot")
    plt.xlabel("Bounding boxes over X-axis")
    plt.ylabel("Y-axis (height)")
    plt.show()

    # Show point cloud with road topology estimation
    z_filtered_pcd = o3d.geometry.PointCloud()
    z_filtered_pcd.points = o3d.utility.Vector3dVector(z_filtered_pc_array)
    o3d.visualization.draw_geometries([z_filtered_pcd, line_set])




"""
if histogram_pc:
    # Convert open3D point cloud to numpy array
    pc_array = np.asarray(pcd.points)

    # FIRST: Cut in z-direction - Get points from sliding window (bounding box)
    for j in range(int(num_iterations_z)):
        # Z-Axis: Define bounds of sliding window
        min_z = z_bound
        max_z = z_bound + hist_window_size

        bb_filter = bounding_box(pc_array, min_x=-np.inf, max_x=np.inf, min_y=-np.inf, max_y=np.inf, min_z=min_z, max_z=max_z)
        z_filtered_pc_array = pc_array[bb_filter]
    
        # Move window
        z_bound = z_bound + hist_window_size        
    
        # Initialize
        x_bound = np.min(pc_array[:, 0])
    
        # X-direction: Calculate number of iterations
        max_value = np.max(pc_array[:, 0])
        num_iterations_x = abs((abs(math.ceil(max_value)) - abs(math.ceil(x_bound)))) / hist_window_size
    
        # SECOND: Cut in x-direction - Get points from sliding window (bounding box)
        for i in range(int(num_iterations_x)):
            # X-Axis: Define bounds of sliding window
            min_x = x_bound
            max_x = x_bound + hist_window_size
    
            # Get points in sliding window (bounding box)
            bb_filter = bounding_box(z_filtered_pc_array, min_x=min_x, max_x=max_x, min_y=-np.inf, max_y=np.inf, min_z=-np.inf, max_z=np.inf)
    
            # Move window
            x_bound = x_bound + hist_window_size
    
            # Calculate the mean, standard deviation and median of each bounding box
            bb_mean = np.mean(z_filtered_pc_array[bb_filter][:, 1])
            bb_std = np.std(z_filtered_pc_array[bb_filter][:, 1])
            bb_median = np.median(z_filtered_pc_array[bb_filter][:, 1])
    
            # Calculate a new coordinate point for this bounding box
            bb_median_coordinates = np.array([[x_bound + hist_window_size/2, bb_median, z_bound + hist_window_size/2]])
    
            # Save new coordinate point to array
            if j == 0:
                est_road_topology = bb_median_coordinates
                lines = [[0, 1]]
            else:
                est_road_topology = np.append(est_road_topology, bb_median_coordinates, axis=0)
                if i != int(num_iterations_x)-1:
                    lines = np.append(lines, [[i, i+1]], axis=0)

    points = est_road_topology

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    z_filtered_pcd = o3d.geometry.PointCloud()
    z_filtered_pcd.points = o3d.utility.Vector3dVector(z_filtered_pc_array)

    o3d.visualization.draw_geometries([z_filtered_pcd, line_set])

"""



