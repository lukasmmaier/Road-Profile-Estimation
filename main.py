# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gin
import os
from absl import app, flags
import open3d as o3d

from preprocessing.preprocess import preprocess_pcd
from utils_and_tools.o3d_tools import estimate_vertex_normals, visualize_keyframe_position
from estimate_road_profile import slice_in_given_dimension


def main(argv):
    # Setup gin-config
    gin.parse_config_files_and_bindings([os.getcwd() + '/configs/config_00.gin'], [])

    # Load point cloud
    pcd = load_point_cloud()

    # Preprocess point cloud data (pcd)
    pcd = preprocess_pcd(pcd=pcd)

    # Estimate vertex normals
    pcd = estimate_vertex_normals(pcd=pcd)

    # Smooth point cloud data (pcd) based on point distribution
    slice_in_given_dimension(pcd)


@gin.configurable
def load_point_cloud(path, filename):
    # Get point cloud path from gin configuration
    filepath = os.path.join(path, filename)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(filepath)

    return pcd


if __name__ == '__main__':
    app.run(main)
