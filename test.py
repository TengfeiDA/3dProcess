import numpy as np
import open3d as o3d
import time
from scipy.spatial.transform import Rotation


def draw_points_cloud(data, window_name="Open3d"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

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


def generate_box_points(length = 10.0, num = 1000):
    xOy = np.zeros((num, 3))
    random_x = np.random.rand(num) * length
    random_y = np.random.rand(num) * length
    xOy[:, 0] = random_x
    xOy[:, 1] = random_y


    xOz = np.zeros((num, 3))
    random_x = np.random.rand(num) * length
    random_z = np.random.rand(num) * length
    xOz[:, 0] = random_x
    xOz[:, 2] = random_z

    yOz = np.zeros((num, 3))
    random_y = np.random.rand(num) * length
    random_z = np.random.rand(num) * length
    yOz[:, 1] = random_y
    yOz[:, 2] = random_z

    points = np.vstack((xOy, yOz, xOz))
    return points


def visualize_pc_pair(src_np, dst_np):
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_np)
    pcd_src.paint_uniform_color([1, 0, 0])

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(dst_np)
    pcd_dst.paint_uniform_color([0, 1, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
    render_option.point_size = 2.0  # 设置点云显示尺寸，尺寸越大，点显示效果越粗
    render_option.show_coordinate_frame = True  # 显示坐标系

    vis.add_geometry(pcd_src)
    vis.add_geometry(pcd_dst)

    vis.run()  # 显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window()  # 销毁窗口，该函数必须从主线程调用


rotation = Rotation.from_euler('xyz', [0.1, 0.2, 0.3])
t = np.asarray([[10],[20],[30]])

raw_box = generate_box_points()
des_box = np.dot(rotation.as_matrix(), raw_box.transpose()) + t
des_box = des_box.transpose()
print(raw_box.shape)
print(des_box.shape)
visualize_pc_pair(raw_box, des_box)
