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



def build_kd_tree(points):
    point_pcd = o3d.geometry.PointCloud()
    point_pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(point_pcd)
    return kdtree

def get_trans_src_points(src_points, r, t):
    src_trans = np.dot(r, src_points.transpose()) + t
    return src_trans.transpose()
    

def get_correspondences(src_points, des_points, des_tree):
    dist = np.zeros(len(src_points))
    corr_i = np.arange(0, len(src_points))
    corr_j = np.zeros(len(src_points))
    for i, point in enumerate(src_points):
        [_, idx, _] = des_tree.search_knn_vector_3d(point, 1)
        j = idx[0]
        corr_j[i] = j
        dist[i] = np.linalg.norm(point - des_points[j])
    corr_i = np.where(dist<10.0)[0]
    corr_j = corr_j[corr_i].astype(int)
    print("get_correspondences:", len(corr_i))
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
    A_cols[0] = (np.multiply(des_normals[corr_j, 2], src_trans[corr_i, 1]) - np.multiply(des_normals[corr_j, 1], src_trans[corr_i, 2])).reshape(-1, 1)
    A_cols[1] = (np.multiply(des_normals[corr_j, 0], src_trans[corr_i, 2]) - np.multiply(des_normals[corr_j, 2], src_trans[corr_i, 0])).reshape(-1, 1)
    A_cols[2] = (np.multiply(des_normals[corr_j, 1], src_trans[corr_i, 2]) + np.multiply(des_normals[corr_j, 0], src_trans[corr_i, 1])).reshape(-1, 1)
    A_cols[3] = (des_normals[corr_j, 0]).reshape(-1, 1)
    A_cols[4] = (des_normals[corr_j, 1]).reshape(-1, 1)
    A_cols[5] = (des_normals[corr_j, 2]).reshape(-1, 1)
    A = np.hstack((A_cols))

    b = np.zeros(len(corr_i))
    for i in range(3):
        b = b + np.multiply(des_normals[corr_j, i], des_points[corr_j, i]) - np.multiply(des_normals[corr_j, 0], src_trans[corr_i, i]) 
    b = b.reshape(-1, 1)
    return A, b


def ICP(src_points, src_normals, des_points, des_normals):
    des_tree = build_kd_tree(des_points)

    r = np.eye(3)
    t = np.zeros((3,1))

    for iteration in range(5):
        print("\niteration", iteration)
        # 1. data association(find nearest point)
        src_trans = get_trans_src_points(src_points, r, t)
        corr_i, corr_j = get_correspondences(src_trans, des_points, des_tree)
        print("2-corr_i:",corr_i[0:10],"\n2-corr_j:",corr_j[0:10])

        # 2. minimize cost function to get R and t (least square)
        A,b = compute_A_b(src_points, des_points, des_normals, corr_i, corr_j)
        tmp = np.linalg.inv(np.matmul(A.transpose(), A))
        x = np.matmul(np.matmul(tmp, A.transpose()), b)
        x = np.squeeze(x)

        # 3. check if converge
        print("x", x)
        rotation = Rotation.from_euler('xyz', x[0:3])
        print("rotation\n",rotation.as_matrix())
        r = rotation.as_matrix()
        t = x[3:6].reshape(-1 ,1)

        src_trans = get_trans_src_points(src_points, r, t)
        corr_i, corr_j = get_correspondences(src_trans, des_points, des_tree)
        print("2-corr_i:",corr_i[0:10],"\n2-corr_j:",corr_j[0:10])
        loss = point_to_plane_loss(src_points, corr_i, des_points, des_normals, corr_j)
        print("loss", loss)

    return


def downsampling(points, normals):
    # print("Before downsampling:", points.shape[0])
    indices = np.arange(0, points.shape[0], 5)
    # print("After downsampling:", len(indices))
    return points[indices], normals[indices]


def main():
    np.set_printoptions(suppress=True)

    data_path = "registration_dataset/point_clouds/"
    files = os.listdir(data_path)

    # 1. 从reg_result.txt中获取source points cloud和destination points cloud
    reg_list = read_reg_results(
        'registration_dataset/reg_result.txt', splitter=',')
    for i in range(1, len(reg_list)):
        src_idx, des_idx, gt_t, gt_rot = reg_result_row_to_array(reg_list[i])

        src_filename = os.path.join(data_path, '%d.bin' % src_idx)
        des_filename = os.path.join(data_path, '%d.bin' % des_idx)
        print("src_filename", src_filename)
        print("des_filename", des_filename)
        src_points, src_normals = read_oxford_bin(src_filename)
        des_points, des_normals = read_oxford_bin(des_filename)

        r = gt_rot.as_matrix()
        t = gt_t.reshape(-1 ,1)
        print("r:\n",r)
        print("t:",t)
        src_tree = build_kd_tree(src_points)
        des_trans = get_trans_src_points(des_points, r, t)
        corr_i, corr_j = get_correspondences(des_trans, src_points, src_tree)
        loss = point_to_plane_loss(des_trans, corr_i, src_points, src_normals, corr_j)
        print("Before downsampling: loss", loss, "\n")

        # draw_points_cloud(src_points, "before downsampling: src points")
        # draw_points_cloud(des_points, "before downsampling: des points")
        # print(src_points.shape)
        # print(des_points.shape)

        # 2. 点云降采样 hash map
        src_points, src_normals = downsampling(src_points, src_normals)
        des_points, des_normals = downsampling(des_points, des_normals)

        src_tree = build_kd_tree(src_points)
        des_trans = get_trans_src_points(des_points, r, t)
        corr_i, corr_j = get_correspondences(des_trans, src_points, src_tree)
        loss = point_to_plane_loss(des_trans, corr_i, src_points, src_normals, corr_j)
        print("After downsampling: loss", loss, "\n")

        # draw_points_cloud(src_points, "after downsampling: src points")

        # 3. ICP
        ICP(src_points, src_normals, des_points, des_normals)
        break


if __name__ == '__main__':
    main()
