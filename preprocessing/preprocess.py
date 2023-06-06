import gin
import open3d as o3d

from preprocessing.preprocess_tools import rotate_pcd, crop_pcd, paint_uniform_pcd
from utils_and_tools.o3d_tools import visualize_keyframe_position


@gin.configurable
def preprocess_pcd(pcd, rotate, crop, paint_uniform, vis_before_after, show_keyframe_position):
    """ Preprocess given point cloud data (pcd). """
    # Rotate point cloud
    if rotate:
        pcd = rotate_pcd(pcd)

    # Visualize point cloud data before cropping or painting
    if vis_before_after:
        vis = [pcd]

        # Generates a box model at the location of a given keyframe
        if show_keyframe_position:
            kf_pos = visualize_keyframe_position()
            vis = [pcd, kf_pos]

    o3d.visualization.draw_geometries(vis)

    # Crop point cloud
    if crop:
        pcd = crop_pcd(pcd)

    # Paint point cloud in uniform color
    if paint_uniform:
        pcd = paint_uniform_pcd(pcd)

    # Visualize point cloud data after cropping or painting
    if vis_before_after:
        vis = [pcd]

        # Generates a box model at the location of a given keyframe
        if show_keyframe_position:
            kf_pos = visualize_keyframe_position()
            vis = [pcd, kf_pos]

        o3d.visualization.draw_geometries(vis)

    return pcd
