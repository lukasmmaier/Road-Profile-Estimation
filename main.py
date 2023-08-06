import gin
import os
from absl import app, flags
import open3d as o3d

import extract_road
from preprocessing.evaluation_processing import process_for_evaluation
from utils_and_tools.o3d_tools import rotate_pcd, remove_points_within_radius

FLAGS = flags.FLAGS
# Global variable
flags.DEFINE_boolean('evaluation', True, 'Specify weather to run algorithm in evaluation mode or not.')


def main(argv):
    # Setup gin-config
    gin.parse_config_files_and_bindings([os.getcwd() + '/configs/config.gin'], [])

    pcd_filename, pcd_gt_filename = load_configuration()

    # Load point cloud from MonoRec
    pcd = load_point_cloud(filename=pcd_filename)

    # Load ground truth (GT) point cloud
    #pcd_gt = load_point_cloud(filename=pcd_gt_filename)

    # Post processing module => Roadway extraction + Denonoising
    road_profile_reconstruction, center = extract_road.extract_road(pcd)

    # Rotate estimates 180° - MonoRec outputs estimates upside down
    road_profile_reconstruction = rotate_pcd(road_profile_reconstruction, [180, 0, 0])
    pcd = rotate_pcd(pcd, [180, 0, 0])

    # Visualize estimate
    o3d.visualization.draw_geometries([road_profile_reconstruction])

    # Further visualizations for evaluation
    if FLAGS.evaluation:
        angle_x, angle_z = load_configuration_for_evaluation()

        # Rotate pcd if desired and custom colorize the height information
        pcd = rotate_pcd(pcd, [angle_x, 0, angle_z], center)
        road_profile_reconstruction, pcd = process_for_evaluation(road_profile_reconstruction, pcd,
                                                                  [angle_x, 0, angle_z], center)

        # Remove road profile points from MonoRec pcd and input post processing pcd for visualization
        pcd = remove_points_within_radius(pcd, road_profile_reconstruction)
        o3d.visualization.draw_geometries([pcd, road_profile_reconstruction])


@gin.configurable
def load_configuration(pcd_filename, pcd_gt_filename, use_gt):
    """Load variables from gin-config.
    Args:
        pcd_filename (str): gin-configurable - Filename for input point cloud (= MonoRec output as .ply).
        pcd_gt_filename (str): gin-configurable - Filename for ground truth point cloud.
        use_gt (bool): gin-configurable - Variable defining, whether ground truth point cloud is used.

    Returns:
        pcd_filename: Filename for input point cloud (= MonoRec output as .ply).
        pcd_gt_filename: Filename for ground truth point cloud.
    """
    # Ground truth (gt) for evaluation (if available)
    if not use_gt:
        pcd_gt_filename = False

    return pcd_filename, pcd_gt_filename


@gin.configurable
def load_point_cloud(filename, pcd_path):
    """Load point cloud data (pcd) using open3D.
    Args:
        filename (str): Filename referring to point cloud file.
        pcd_path (str): gin-configurable - Path where point clouds are stored.

    Returns:
        pcd: Open3D Point Cloud Data (PCD).
    """
    path = os.path.dirname(os.getcwd())
    path = os.path.join(path, pcd_path)

    if filename != False:
        # Load point cloud
        filepath = os.path.join(path, filename)
        pcd = o3d.io.read_point_cloud(filepath)
    else:
        pcd = False

    return pcd

@gin.configurable
def load_configuration_for_evaluation(angle_x, angle_z):
    """Load variables from gin-config.
    Args/Returns:
        angle_x (float): gin-configurable - Angle (in degrees) for the rotation of the pcd around the x-axis.
        angle_z (float): gin-configurable - Angle (in degrees) for the rotation of the pcd around the z-axis.
    """
    return angle_x, angle_z


if __name__ == '__main__':
    app.run(main)
