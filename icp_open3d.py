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
    corr_i = np.where(dist < 5.0)[0]
    corr_j = corr_j[corr_i].astype(int)
    print("correspondences:", len(corr_i))
    return corr_i, corr_j


def point_to_plane_loss(src_points, corr_i, des_points, des_normals, corr_j):
    point_diff = src_points[corr_i] - des_points[corr_j]
    loss = 0
    for i, diff in enumerate(point_diff):
        tmp = np.dot(diff, des_normals[corr_j[i]].transpose())
        loss = loss + tmp*tmp
    return loss


def compute_A_b(src_trans, des_points, des_normals, corr_i, corr_j):
    A_cols = [[]] * 6
    A_cols[0] = (np.multiply(des_normals[corr_j, 2], src_trans[corr_i, 1]) -
                 np.multiply(des_normals[corr_j, 1], src_trans[corr_i, 2])).reshape(-1, 1)
    A_cols[1] = (np.multiply(des_normals[corr_j, 0], src_trans[corr_i, 2]) -
                 np.multiply(des_normals[corr_j, 2], src_trans[corr_i, 0])).reshape(-1, 1)
    A_cols[2] = (np.multiply(des_normals[corr_j, 1], src_trans[corr_i, 2]) -
                 np.multiply(des_normals[corr_j, 0], src_trans[corr_i, 1])).reshape(-1, 1)
    A_cols[3] = (des_normals[corr_j, 0]).reshape(-1, 1)
    A_cols[4] = (des_normals[corr_j, 1]).reshape(-1, 1)
    A_cols[5] = (des_normals[corr_j, 2]).reshape(-1, 1)
    A = np.hstack((A_cols))

    b = np.zeros(len(corr_i))
    for i in range(3):
        b = b + np.multiply(des_normals[corr_j, i], des_points[corr_j, i]) - \
            np.multiply(des_normals[corr_j, i], src_trans[corr_i, i])
    b = b.reshape(-1, 1)
    return A, b


def ICP(src_points, src_normals, des_points, des_normals, gt_r, gt_t, gt_loss):
    # des_tree = build_kd_tree(des_points)

    r = np.eye(3)
    t = np.zeros((3, 1))
    src_trans = get_trans_src_points(src_points, r, t)
    visualize_pc_pair(src_trans, des_points)

    for iteration in range(50):
        print("\niteration", iteration)
        # 1. data association(find nearest point)
        src_trans = get_trans_src_points(src_points, r, t)
        corr_i, corr_j = get_correspondences(src_trans, des_points)
        loss = point_to_plane_loss(
            src_points, corr_i, des_points, des_normals, corr_j)

        rotation = Rotation.from_matrix(r)
        print("r: ", rotation.as_quat())
        print("gt:", gt_r.as_quat())
        print("t: ", t.squeeze())
        print("gt:", gt_t.squeeze())
        print("loss / gt:", round(loss, 2), "/", round(gt_loss, 2))

        # 2. minimize cost function to get R and t (least square)
        A, b = compute_A_b(src_points, des_points, des_normals, corr_i, corr_j)
        tmp = np.linalg.inv(np.matmul(A.transpose(), A))
        x = np.matmul(np.matmul(tmp, A.transpose()), b)
        x = np.squeeze(x)
        print(x)
        print(np.sin(x[1]))
        print(np.cos(x[1]))

        # 3. check if converge
        rotation = Rotation.from_euler('xyz', x[0:3])
        r = rotation.as_matrix()
        t = x[3:6].reshape(-1, 1)

    src_trans = get_trans_src_points(src_points, r, t)
    corr_i, corr_j = get_correspondences(src_trans, des_points)
    loss = point_to_plane_loss(
        src_points, corr_i, des_points, des_normals, corr_j)

    print("\nFinal result:")
    rotation = Rotation.from_matrix(r)
    print("r: ", rotation.as_quat())
    print("gt:", gt_r.as_quat())
    print("t: ", t.squeeze())
    print("gt:", gt_t.squeeze())
    print("loss / gt:", round(loss, 2), "/", round(gt_loss, 2))

    src_trans = get_trans_src_points(src_points, r, t)
    visualize_pc_pair(src_trans, des_points)

    return


def downsampling(points, normals):
    # print("Before downsampling:", points.shape[0])
    indices = np.arange(0, points.shape[0], 2)
    # print("After downsampling:", len(indices))
    return points[indices], normals[indices]


def main():
    np.set_printoptions(precision=5, suppress=True)

    data_path = "registration_dataset/point_clouds/"
    files = os.listdir(data_path)

    # 1. 从reg_result.txt中获取source points cloud和destination points cloud
    reg_list = read_reg_results(
        'registration_dataset/reg_result.txt', splitter=',')
    for i in range(1, len(reg_list)):
        des_idx, src_idx, gt_t, gt_rot = reg_result_row_to_array(reg_list[i])

        src_filename = os.path.join(data_path, '%d.bin' % src_idx)
        des_filename = os.path.join(data_path, '%d.bin' % des_idx)
        print("src_filename", src_filename)
        print("des_filename", des_filename)
        src_points, src_normals = read_oxford_bin(src_filename)
        des_points, des_normals = read_oxford_bin(des_filename)

        print("Before downsampling: src points num:", src_points.shape[0])
        print("Before downsampling: des points num:", des_points.shape[0])

        # 2. 点云降采样 hash map
        src_points, src_normals = downsampling(src_points, src_normals)
        des_points, des_normals = downsampling(des_points, des_normals)

        print("After downsampling: src points num:", src_points.shape[0])
        print("After downsampling: des points num:", des_points.shape[0])

        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(src_points)
        pcd_src.estimate_normals()

        pcd_des = o3d.geometry.PointCloud()
        pcd_des.points = o3d.utility.Vector3dVector(des_points)
        pcd_des.estimate_normals()

        # 3. ICP
        icp = o3d.pipelines.registration.registration_icp(
            source=pcd_src, target=pcd_des,
            max_correspondence_distance=10.0,    # 距离阈值
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

        print(icp)

        pcd_src.transform(icp.transformation)
        src_trans = np.array(pcd_src.points)

        corr_i, corr_j = get_correspondences(src_trans, des_points)
        gt_loss = point_to_plane_loss(
            src_trans, corr_i, des_points, src_normals, corr_j)
        print("gt_loss", gt_loss)

        visualize_pc_pair(src_trans, des_points)
        break


if __name__ == '__main__':
    main()
