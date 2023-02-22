import numpy as np
import open3d as o3d
import time
import math


def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4):
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)
        m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        for i in range(m):
            value = f.readline().rstrip().split(' ')
            faces.append([int(x) for x in value])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces


def draw_visualize(point_cloud, line_pcd=None, sphere_pcds=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
    render_option.point_size = 2.0  # 设置点云显示尺寸，尺寸越大，点显示效果越粗
    render_option.show_coordinate_frame = True  # 显示坐标系

    vis.add_geometry(point_cloud)
    if line_pcd:
        vis.add_geometry(line_pcd)
    if sphere_pcds:
        for sphere_pcd in sphere_pcds:
            vis.add_geometry(sphere_pcd)

    vis.run()  # 显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window()  # 销毁窗口，该函数必须从主线程调用


def get_points_cloud_pcd(points, all_indices):
    # input: points, np.array of [n,3]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.paint_uniform_color([1,1,1])

    colors = np.ones((points.shape[0], 3))
    for indices in all_indices:
        colors[indices] = np.asarray([1, 0, 0])
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def get_sphere_pcd(center, r):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r/2)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([1, 0.1, 0.1])
    vertices = np.asarray(mesh_sphere.vertices) + center
    mesh_sphere.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh_sphere


def build_kd_tree_and_estimate_normals(points):
    point_pcd = o3d.geometry.PointCloud()
    point_pcd.points = o3d.utility.Vector3dVector(points)
    point_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(point_pcd.normals)
    kdtree = o3d.geometry.KDTreeFlann(point_pcd)
    return kdtree, normals


def get_neighbor_points(kdtree, points, query_index, radius):
    [_, idx, _] = kdtree.search_radius_vector_3d(
        points[query_index], radius)
    neighbor_indices = np.asarray(idx)
    dist = np.linalg.norm(
        points[neighbor_indices] - points[query_index], axis=1)
    neighbor_indices = np.delete(
        neighbor_indices, np.where(dist < 1e-3))
    return neighbor_indices


def SPFH(key_point, key_point_normal, neighbor_points, neighbor_points_normal, bin_num):
    alpha_bin_size = 2.0 / (bin_num-1)
    phi_bin_size = 2.0 / (bin_num-1)
    theta_bin_size = 2 * np.pi / (bin_num-1)
    alpha_histogram = np.zeros(bin_num)
    phi_histogram = np.zeros(bin_num)
    theta_histogram = np.zeros(bin_num)
    u = key_point_normal
    for i, point in enumerate(neighbor_points):
        n2 = neighbor_points_normal[i]
        p12 = point - key_point
        p12 = p12 / np.linalg.norm(p12)
        v = np.cross(u, p12)
        w = np.cross(u, v)
        alpha = np.dot(v, n2)
        phi = np.dot(u, p12)

        theta = math.atan2(np.dot(w, n2), np.dot(u, n2))
        alpha_bin = int(np.round((alpha + 1.0) / alpha_bin_size))
        phi_bin = int(np.round((phi + 1.0) / phi_bin_size))
        theta_bin = int(np.round((theta + np.pi) / theta_bin_size))

        alpha_histogram[alpha_bin] = alpha_histogram[alpha_bin] + 1
        phi_histogram[phi_bin] = phi_histogram[phi_bin] + 1
        theta_histogram[theta_bin] = theta_histogram[theta_bin] + 1

    return np.concatenate((alpha_histogram, phi_histogram, theta_histogram))


def weighted_sum_of_spfh(key_point, neighbor_points, neighbor_spfhs):
    result = np.zeros(neighbor_spfhs.shape[1])
    for i, point in enumerate(neighbor_points):
        weight = 1.0 / np.linalg.norm(point - key_point)
        result = result + weight * neighbor_spfhs[i]
    result = result / neighbor_points.shape[0]
    return result


def FPFH(points, radius):
    points_num = points.shape[0]

    start_time = time.time()

    kdtree, normals = build_kd_tree_and_estimate_normals(points)

    end_time = time.time()
    print("Build KD-Tree: {:.2f}s".format(end_time - start_time))

    start_time = time.time()

    bin_num = 21
    all_spfhs = np.zeros((points_num, 3*bin_num))
    all_neighbors_indices = [None] * points_num
    for i, point in enumerate(points):
        neighbor_indices = get_neighbor_points(kdtree, points, i, radius)
        if neighbor_indices.shape[0] > 10:
            histogram = SPFH(points[i], normals[i], points[neighbor_indices],
                             normals[neighbor_indices], bin_num)
            all_spfhs[i] = histogram
            all_neighbors_indices[i] = neighbor_indices

    end_time = time.time()
    print("Compute all spfhs: {:.2f}s".format(end_time - start_time))

    start_time = time.time()

    all_fpfhs = np.zeros((points_num, 3*bin_num))
    for i, point in enumerate(points):
        neighbor_indices = all_neighbors_indices[i]
        if neighbor_indices is not None and neighbor_indices.shape[0] > 10:
            all_fpfhs[i] = all_spfhs[i] + weighted_sum_of_spfh(
                point, points[neighbor_indices], all_spfhs[neighbor_indices])

    end_time = time.time()
    print("Compute all fpfhs: {:.2f}s".format(end_time - start_time))


def main():
    off_filename = "./data/airplane_0627.off"
    points, _ = read_off(off_filename)
    min_pos = points.min(axis=0)
    max_pos = points.max(axis=0)
    radius = 0.02 * (max_pos-min_pos).max()
    np.set_printoptions(suppress=True)
    points_num = points.shape[0]
    print("points num: ", points_num)
    print("min_pos: ", min_pos)
    print("max_pos: ", max_pos)
    print("radius: ", radius)

    FPFH(points, radius)


if __name__ == '__main__':
    main()
