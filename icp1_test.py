import os
import time
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
    return points_normals[:, 0:3], points_normals[:, 3:6]


def read_reg_results(file_path, splitter=','):
    reg_gt_list = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(splitter)
            items = [x.strip() for x in items]
            reg_gt_list.append(items)
            line = f.readline()
    return reg_gt_list


def reg_result_row_to_array(reg_result_row):
    idx1 = int(reg_result_row[0])
    idx2 = int(reg_result_row[1])
    t = np.asarray([float(x) for x in reg_result_row[2:5]])
    q_wxyz = [float(x) for x in reg_result_row[5:9]]
    q_xyzw = np.asarray(q_wxyz[1:] + q_wxyz[:1])

    rot = Rotation.from_quat(q_xyzw)

    return idx1, idx2, t, rot


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


def visualize_correspondence(src_np, dst_np, corr_i, corr_j):
    points = np.vstack((src_np, dst_np))
    num = src_np.shape[0]
    src_colors = np.tile(np.asarray([1, 0, 0]), (num, 1))
    des_colors = np.tile(np.asarray([0, 1, 0]), (num, 1))
    colors = np.vstack((src_colors, des_colors))

    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points)
    points_pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
    render_option.point_size = 2.0  # 设置点云显示尺寸，尺寸越大，点显示效果越粗
    render_option.show_coordinate_frame = True  # 显示坐标系

    vis.add_geometry(points_pcd)

    lines = np.hstack((corr_i.reshape(-1, 1), corr_j.reshape(-1, 1)+num))
    print(lines.shape)
    colors = np.tile(np.asarray([0, 0, 1]), (lines.shape[0], 1))
    line_pcd = o3d.geometry.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(colors)
    line_pcd.points = o3d.utility.Vector3dVector(points)

    vis.add_geometry(line_pcd)

    vis.run()  # 显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window()  # 销毁窗口，该函数必须从主线程调用


def build_kd_tree(points):
    point_pcd = o3d.geometry.PointCloud()
    point_pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(point_pcd)
    return kdtree


def get_trans_src_points(src_points, r, t):
    src_trans = np.dot(r, src_points.transpose()) + t
    return src_trans.transpose()


def get_nearest_point(query, points):
    dist = np.linalg.norm(query - points, axis=1)
    idx = np.argmin(dist)
    return idx, dist[idx]


def get_correspondences(src_points, des_points, des_tree=None):
    dist = np.zeros(len(src_points))
    corr_i = np.arange(0, len(src_points))
    corr_j = np.zeros(len(src_points))
    for i, point in enumerate(src_points):
        # [_, idx, _] = des_tree.search_knn_vector_3d(point, 1)
        min_idx, min_dist = get_nearest_point(point, des_points)
        corr_j[i] = min_idx
        dist[i] = min_dist
    corr_i = np.where(dist < 100.0)[0]
    corr_j = corr_j[corr_i].astype(int)
    print("correspondences:", len(corr_i))
    return corr_i, corr_j


def ICP(src_points, des_points):
    # des_tree = build_kd_tree(des_points)

    r = np.eye(3)
    t = np.zeros((3, 1))

    for iteration in range(20):
        print("\niteration", iteration)
        # 1. data association(find nearest point)
        src_trans = get_trans_src_points(src_points, r, t)
        corr_i, corr_j = get_correspondences(src_trans, des_points)
        visualize_correspondence(src_trans, des_points, corr_i, corr_j)

        loss = np.linalg.norm(
            src_points[corr_i] - des_points[corr_j], axis=1).sum()

        rotation = Rotation.from_matrix(r)
        print("r: ", rotation.as_quat())
        print("t: ", t.squeeze())
        print("loss:", round(loss, 2))

        # 2. minimize cost function to get R and t (least square)
        src_mean = src_points[corr_i].mean(axis=0)
        des_mean = des_points[corr_j].mean(axis=0)
        src_corr_points = src_points[corr_i] - src_mean
        des_corr_points = des_points[corr_j] - des_mean
        tmp = np.matmul(des_corr_points.transpose(), src_corr_points)
        U, sigma, VT = np.linalg.svd(tmp)
        r = np.matmul(U, VT)
        t = (des_mean - np.matmul(r, src_mean)).reshape(-1, 1)

        # 3. check if converge
        # rotation = Rotation.from_euler('xyz', x[0:3])
        # r = rotation.as_matrix()
        # t = x[3:6].reshape(-1, 1)

    src_trans = get_trans_src_points(src_points, r, t)
    corr_i, corr_j = get_correspondences(src_trans, des_points)
    loss = np.linalg.norm(
        src_points[corr_i] - des_points[corr_j], axis=1).sum()

    print("\nFinal result:")
    rotation = Rotation.from_matrix(r)
    print("r: ", rotation.as_quat())
    print("t: ", t.squeeze())
    print("loss:", round(loss, 2))

    src_trans = get_trans_src_points(src_points, r, t)
    visualize_correspondence(src_trans, des_points, corr_i, corr_j)

    return


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
    

def main():
    np.set_printoptions(suppress=True)

    rotation = Rotation.from_euler('xyz', [0.2, 0.5, 0.8])
    t = np.asarray([[10],[20],[30]])

    src_box = generate_box_points()
    des_box = np.dot(rotation.as_matrix(), src_box.transpose()) + t
    des_box = des_box.transpose()

    # 3. ICP
    ICP(src_box, des_box)


if __name__ == '__main__':
    main()
