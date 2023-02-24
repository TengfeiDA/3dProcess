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


def ICP(src_points, src_normals, des_points, des_normals):
    # 1. data association(find nearest point)

    # 2. minimize cost function to get R and t (least square)

    # 3. check if converge

    return


def downsampling(points, normals):
    indices = np.arange(0, points.shape[0], 5)
    return points[indices], normals[indices]


def main():

    data_path = "registration_dataset/point_clouds/"
    files = os.listdir(data_path)

    # 1. 从reg_result.txt中获取source points cloud和destination points cloud
    reg_list = read_reg_results(
        'registration_dataset/reg_result.txt', splitter=',')
    for i in range(1, len(reg_list)):
        src_idx, des_idx, t, rot = reg_result_row_to_array(reg_list[i])

        src_points, src_normals = read_oxford_bin(
            os.path.join(data_path, '%d.bin' % src_idx))
        des_points, des_normals = read_oxford_bin(
            os.path.join(data_path, '%d.bin' % des_idx))
        draw_points_cloud(src_points, "before downsampling: src points")
        # draw_points_cloud(des_points, "before downsampling: des points")
        print(src_points.shape)
        # print(des_points.shape)

        # 2. 点云降采样 hash map
        src_points, src_normals = downsampling(src_points, src_normals)
        des_points, des_normals = downsampling(des_points, des_normals)

        draw_points_cloud(src_points, "after downsampling: src points")
        print(src_points.shape)

        # 3. ICP
        break


if __name__ == '__main__':
    main()
