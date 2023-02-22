import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
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


# def draw_visualize(point_cloud, line_pcd=None, sphere_pcds=None):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     render_option = vis.get_render_option()
#     render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色
#     render_option.point_size = 2.0  # 设置点云显示尺寸，尺寸越大，点显示效果越粗
#     render_option.show_coordinate_frame = True  # 显示坐标系

#     vis.add_geometry(point_cloud)
#     if line_pcd:
#         vis.add_geometry(line_pcd)
#     if sphere_pcds:
#         for sphere_pcd in sphere_pcds:
#             vis.add_geometry(sphere_pcd)

#     vis.run()  # 显示窗口，会阻塞当前线程，直到窗口关闭
#     vis.destroy_window()  # 销毁窗口，该函数必须从主线程调用

def draw_visualize(point_cloud, line_pcd=None, sphere_pcds=None, feature_indices=None):
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True

    if line_pcd is not None:
        vis.add_geometry("line", line_pcd)
    if sphere_pcds is not None:
        for i, sphere_pcd in enumerate(sphere_pcds):
            vis.add_geometry("sphere-{}".format(i), sphere_pcd)

    if feature_indices is not None:
        for i, idx in enumerate(feature_indices):
            vis.add_3d_label(point_cloud.points[idx], "{}".format(i))

    vis.add_geometry("points", point_cloud)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def get_points_cloud_pcd(points, all_indices=[]):
    # input: points, np.array of [n,3]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.paint_uniform_color([1,1,1])

    colors = np.zeros((points.shape[0], 3))
    for indices in all_indices:
        colors[indices] = np.asarray([1, 0, 0])
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def get_eigenvectors_pcd(point, eigenvectors):
    end1 = point + np.squeeze(np.asarray(eigenvectors[:, 2])*100)
    end2 = point + np.squeeze(np.asarray(eigenvectors[:, 1])*60)
    end3 = point + np.squeeze(np.asarray(eigenvectors[:, 0])*30)
    line_points = np.vstack((point, end1, end2, end3))

    line_pcd = o3d.geometry.LineSet()
    line_pcd.points = o3d.utility.Vector3dVector(line_points)
    line_pcd.lines = o3d.utility.Vector2iVector(
        np.asarray([[0, 1], [0, 2], [0, 3]]))
    line_pcd.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return line_pcd


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


def get_points_weights(points, r):
    weights = np.zeros(points.shape[0])
    for i, point in enumerate(points):
        diff = points - point
        dist = np.linalg.norm(diff, axis=1)
        neighbors_num = (dist <= r).sum()
        weights[i] = 1 / neighbors_num
    return weights


def PCA(points, query):
    points = points - query
    H = np.matmul(points.transpose(), points)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    sort_index = eigenvalues.argsort()
    eigenvalues = eigenvalues[sort_index]
    eigenvectors = eigenvectors[:, sort_index]
    print("eigenvalues", eigenvalues)
    print("eigenvectors\n", eigenvectors)

    return eigenvalues, eigenvectors


def WPCA(points, query_point, weights):
    points = points - query_point
    weights_matrix = np.diag(weights)
    H = np.matmul(np.matmul(points.transpose(), weights_matrix), points)
    H = H / weights.sum()
    eigenvalues, _ = np.linalg.eig(H)
    eigenvalues = np.sort(eigenvalues)
    return eigenvalues


def ISS(points, all_neighbors_indices, radius, gamma21_gating, gamma32_gating):
    start_time = time.time()
    # 0. 计算每个点的R邻域内点的数量，在后面计算权重用
    weights = get_points_weights(points, radius)
    pnum = points.shape[0]
    gamma21 = np.zeros(pnum)
    gamma32 = np.zeros(pnum)
    min_eigenvalue = np.zeros(pnum)
    is_feature = np.zeros(pnum, dtype=bool)

    for query_index, query_point in enumerate(points):
        if query_index % 5000 == 0:
            end_time = time.time()
            print("query_index:", query_index, "/", pnum,
                  " time: {:.2f}s".format(end_time - start_time))

        # 1. 对每个点选取周围r半径内的邻域点
        neighbor_indices = all_neighbors_indices[query_index]
        if len(neighbor_indices) <= 0:
            continue

        # 2. 对每个邻域内的所有点，一起计算加权PCA
        eigenvalues = WPCA(points[neighbor_indices],
                           query_point, weights[neighbor_indices])

        # 3. 记录最小特征值大于0的结果
        min_eigenvalue[query_index] = abs(eigenvalues[0])
        if min_eigenvalue[query_index] < 1e-3:
            continue

        gamma21[query_index] = eigenvalues[1] / eigenvalues[0]
        gamma32[query_index] = eigenvalues[2] / eigenvalues[1]

        # 4. 根据特征值之间比例关系判断是否为特征点
        if gamma21[query_index] > gamma21_gating and gamma32[query_index] > gamma32_gating:
            is_feature[query_index] = True

    # NMS
    print("\nBefore NMS, features num:", is_feature.sum())
    for query_index, neighbor_indices in enumerate(all_neighbors_indices):
        if is_feature[query_index] == False:
            continue
        for index in neighbor_indices:
            if is_feature[index] == True and min_eigenvalue[index] < min_eigenvalue[query_index]:
                is_feature[index] = False
    print("After NMS, features num:", is_feature.sum())

    feature_indices = np.where(is_feature == True)[0]

    end_time = time.time()
    print("ISS: {:.2f}s".format(end_time - start_time))

    return feature_indices


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


def weighted_sum_of_spfh(points, all_neighbors_indices, all_normals, bin_num, query_index):
    neighbor_indices = all_neighbors_indices[query_index]
    neighbors_num = len(neighbor_indices)
    neighbor_spfhs = np.zeros((neighbors_num, 3*bin_num))
    result = np.zeros(neighbor_spfhs.shape[1])
    for i, index in enumerate(neighbor_indices):
        neighbor_neighbor_indices = all_neighbors_indices[index]
        neighbor_spfhs = SPFH(points[index], all_normals[index], points[neighbor_neighbor_indices],
                              all_normals[neighbor_neighbor_indices], bin_num)
        weight = 1.0 / np.linalg.norm(points[index] - points[query_index])
        result = result + weight * neighbor_spfhs

    result = result / neighbors_num
    return result


def FPFH(points, all_neighbors_indices, all_normals, feature_indices):
    start_time = time.time()

    bin_num = 11
    all_fpfhs = np.zeros((len(feature_indices), 3*bin_num))
    for i, index in enumerate(feature_indices):
        neighbor_indices = all_neighbors_indices[index]
        self_spfh = SPFH(points[index], all_normals[index], points[neighbor_indices],
                         all_normals[neighbor_indices], bin_num)
        neighbors_spfh = weighted_sum_of_spfh(
            points, all_neighbors_indices, all_normals, bin_num, index)
        all_fpfhs[i] = self_spfh + neighbors_spfh

    end_time = time.time()
    print("FPFH: {:.2f}s".format(end_time - start_time))

    return all_fpfhs


def compare(vectors):
    for i, p in enumerate(vectors):
        dist = np.linalg.norm(vectors - p, axis=1)
        dist[i] = np.finfo(np.float32).max
        sort_index = dist.argsort()
        print("i =", i, " -", sort_index[0], "top 3 nearest index:",
              sort_index[0:3], "distance:", dist[sort_index[0:3]])


def delete_same_points(points):
    delete_indices = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        if delete_indices[i] == False:
            diff = points - point
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = np.finfo(np.float32).max
            dist[delete_indices] = np.finfo(np.float32).max
            indices = np.where(dist <= 1e-3)
            delete_indices[indices[0]] = True
    return points[~delete_indices]


def main():
    off_filename = "./data/airplane_0628.off"
    points, _ = read_off(off_filename)

    np.set_printoptions(suppress=True)

    print("\nbefore delete:")
    points_num = points.shape[0]
    print("points num: ", points_num)

    points = delete_same_points(points)

    print("\nafter delete:")
    points_num = points.shape[0]
    min_pos = points.min(axis=0)
    max_pos = points.max(axis=0)
    print("points num: ", points_num)
    print("min_pos: ", min_pos)
    print("max_pos: ", max_pos)

    radius = 0.02 * (max_pos-min_pos).max()
    print("\nradius: ", radius)

    start_time = time.time()
    kdtree, all_normals = build_kd_tree_and_estimate_normals(points)
    end_time = time.time()
    print("\nBuild KD-Tree: {:.2f}s".format(end_time - start_time))

    start_time = time.time()
    all_neighbors_indices = [np.asarray([])] * points_num
    for i in range(points_num):
        all_neighbors_indices[i] = get_neighbor_points(
            kdtree, points, i, radius)
    end_time = time.time()
    print("\nGet all neighbors: {:.2f}s".format(end_time - start_time))

    gamma21_gating = 5
    gamma32_gating = 10
    print("\ngamma21_gating", gamma21_gating)
    print("gamma32_gating", gamma32_gating)

    feature_indices = ISS(points, all_neighbors_indices,
                          radius, gamma21_gating, gamma32_gating)
    # print("\nAll feature points")
    # for i, index in enumerate(feature_indices):
    #     print(i, ",", points[index])
    # print("\nCompare feature points")
    # compare(points[feature_indices])

    all_fpfhs = FPFH(points, all_neighbors_indices,
                     all_normals, feature_indices)
    # print("\nAll feature histograms")
    # for i, fpfh in enumerate(all_fpfhs):
    #     print(i, ":", fpfh)
    print("\nCompare feature histograms")
    compare(all_fpfhs)

    sphere_pcds = []
    for index in feature_indices:
        sphere_pcd = get_sphere_pcd(points[index], radius)
        sphere_pcds.append(sphere_pcd)

    point_cloud_pcd = get_points_cloud_pcd(points)
    draw_visualize(point_cloud_pcd, sphere_pcds=sphere_pcds,
                   feature_indices=feature_indices)


if __name__ == '__main__':
    main()
