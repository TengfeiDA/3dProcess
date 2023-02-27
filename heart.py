import numpy as np
import open3d as o3d
import time


def draw_points_cloud(data, window_name="Open3d"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
    render_option.point_size = 2.0  # 设置点云显示尺寸，尺寸越大，点显示效果越粗
    render_option.show_coordinate_frame = True  # 显示坐标系

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    point_cloud.paint_uniform_color([1, 0, 0])

    vis.add_geometry(point_cloud)  # 添加显示的点云对象
    # vis.add_geometry(get_grid(data.min(axis=0)[2]-1))

    vis.run()  # 显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window()  # 销毁窗口，该函数必须从主线程调用



grid = np.arange(-1.2, 1.2, 0.01)
gridinv = np.arange(1.2, -1.2, -0.01)

points = []
for x in grid:
    for y in grid:
        gating = 1e-4
        new_points = []
        zs = []
        for z in grid:
            e = (x**2 + y**2 * 9 / 80) * z**3 - (x**2 + y**2*9/4 + z**2 - 1)**3
            if abs(e) < gating:
                new_points.append([x, y, z])
                zs.append(z)
        if len(new_points) > 5:
            tmp = (zs[-1] - zs[0]) / len(zs)
            if tmp < 0.04:
                l = int(len(new_points)/2)
                new_points = [new_points[l]]
        points.extend(new_points)

points = np.asarray(points)
print(points.shape)
draw_points_cloud(points)