import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def read_oxford_bin(bin_path):
    '''
    :param path:
    :return: [x,y,z,nx,ny,nz]: Nx6
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    points_normals = np.reshape(data_np, (int(data_np.shape[0]/6), 6))
    return points_normals[:, 0:3], points_normals[:, 3:5]


def draw_points_cloud(data):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
    render_option.point_size = 2.0  # 设置点云显示尺寸，尺寸越大，点显示效果越粗
    render_option.show_coordinate_frame = True  # 显示坐标系

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    point_cloud.paint_uniform_color([1, 1, 1])

    vis.add_geometry(point_cloud)  # 添加显示的点云对象
    # vis.add_geometry(get_grid(data.min(axis=0)[2]-1))

    vis.run()  # 显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window()  # 销毁窗口，该函数必须从主线程调用


def main():

    data_path = "registration_dataset/point_clouds/"
    files = os.listdir(data_path)
    for file in files:
        filename = data_path + file
        print("filename:", filename)
        points, normals = read_oxford_bin(filename)
        draw_points_cloud(points)


if __name__ == '__main__':
    main()
