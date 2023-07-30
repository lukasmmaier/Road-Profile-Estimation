import gin
import os
from absl import app, flags
import open3d as o3d
import numpy as np

import extract_road
from preprocessing.evaluation_processing import rotate_pcd, process_for_evaluation
from preprocessing.preprocess import preprocess_pcd
from utils_and_tools import o3d_tools
from estimate_road_profile import slice_in_given_dimension


def main(argv):
    # Setup gin-config
    gin.parse_config_files_and_bindings([os.getcwd() + '/configs/config.gin'], [])

    # Load point cloud from MonoRec
    pcd = load_point_cloud(filename="seequence_00_1870-1970.ply")

    # Load GT point cloud
    pcd_gt = load_point_cloud(filename='seequence_05_0-250_gt.ply')

    #pcd = rotate_pcd(pcd, [180, 0, 0])
    #o3d.visualization.draw_geometries([pcd])

    # Preprocess point cloud data (pcd)
    #pcd = preprocess_pcd(pcd=pcd)

    road_profile_reconstruction, center = extract_road.extract_road(pcd, pcd_gt)

    road_profile_reconstruction = rotate_pcd(road_profile_reconstruction, [180, 0, 0])
    pcd = rotate_pcd(pcd, [180, 0, 0])

    angle_x = -1.8
    angle_z = -1.8

    pcd = rotate_pcd(pcd, [angle_x, 0, angle_z], center)

    road_profile_reconstruction, pcd = process_for_evaluation(road_profile_reconstruction, pcd, [angle_x, 0, angle_z], center)

    o3d.visualization.draw_geometries([road_profile_reconstruction])


    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black


    # Remove road points from original pcd and input reconstruction for visualization
    pcd = o3d_tools.remove_points_within_radius(pcd, road_profile_reconstruction, 0.4)
    #o3d.visualization.draw_geometries([pcd, road_profile_reconstruction])

    o3d.visualization.draw_geometries_with_key_callbacks([pcd, road_profile_reconstruction], key_to_callback)


@gin.configurable
def load_point_cloud(filename, path):
    # Get point cloud path from gin configuration
    filepath = os.path.join(path, filename)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(filepath)

    return pcd


if __name__ == '__main__':
    app.run(main)
