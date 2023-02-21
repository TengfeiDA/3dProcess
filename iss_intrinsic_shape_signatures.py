import numpy as np
import open3d as o3d
import time


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


def get_neighbor_points(points, query_index, r):
    diff = points - points[query_index]
    dist = np.linalg.norm(diff, axis=1)
    indices = dist <= r
    indices[query_index] = False
    return indices


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
    # centroid = np.mean(points, axis=0)
    # points = points - centroid
    # print("center", center)
    # print("centroid", centroid)
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


def ISS(points, radius=20):
    start_time = time.time()

    # 0. 计算每个点的R邻域内点的数量，在后面计算权重用
    weights = get_points_weights(points, radius)
    pnum = points.shape[0]
    alpha = np.zeros(pnum)
    beta = np.zeros(pnum)
    min_eigenvalue = np.zeros(pnum)
    is_feature = np.zeros(pnum, dtype=bool)

    all_indices = []
    point_pcd = o3d.geometry.PointCloud()
    point_pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(point_pcd)

    end_time = time.time()
    print("Build KD-Tree: {:.2f}秒".format(end_time - start_time))

    for query_index, query_point in enumerate(points):
        if query_index % 1000 == 0:
            print("query_index:", query_index, "/", pnum)
            if query_index >= 5000:
                break

        # 1. 对每个点选取周围r半径内的邻域点
        [_, idx, _] = pcd_tree.search_radius_vector_3d(
            points[query_index], radius)
        neighbor_indices = np.asarray(idx)
        # neighbor_indices = np.delete(
        #     neighbor_indices, np.where(neighbor_indices == query_index))
        all_indices.append(neighbor_indices)

        # 2. 对每个邻域内的所有点，一起计算加权PCA
        eigenvalues = WPCA(points[neighbor_indices],
                           query_point, weights[neighbor_indices])

        # 3. 记录最小特征值大于0的结果
        min_eigenvalue[query_index] = eigenvalues[0]
        if min_eigenvalue[query_index] < 1e-3:
            continue

        alpha[query_index] = eigenvalues[1] / eigenvalues[0]
        beta[query_index] = eigenvalues[2] / eigenvalues[1]

        # 4. 根据特征值之间比例关系判断是否为特征点
        if alpha[query_index] > 5 and beta[query_index] > 5:
            is_feature[query_index] = True
            # print("query_index:", query_index)
            # print("alpha:", alpha[query_index])
            # print("beta:", beta[query_index])
            # print()

    # NMS
    print("Before NMS, features num:", is_feature.sum())
    for query_index, neighbor_indices in enumerate(all_indices):
        if is_feature[query_index] == False:
            continue
        for index in neighbor_indices:
            if is_feature[index] == True and min_eigenvalue[index] < min_eigenvalue[query_index]:
                is_feature[index] = False
    print("After NMS, features num:", is_feature.sum())

    feature_neighbors_indices = [indices for i, indices in enumerate(
        all_indices) if is_feature[i] == True]
    point_cloud_pcd = get_points_cloud_pcd(points, feature_neighbors_indices)

    sphere_pcds = []
    for i, flag in enumerate(is_feature):
        if flag == True:
            sphere_pcd = get_sphere_pcd(points[i], radius)
            sphere_pcds.append(sphere_pcd)
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))

    draw_visualize(point_cloud_pcd, sphere_pcds=sphere_pcds)

    return


def main():
    off_filename = "./data/airplane_0628.off"
    points, _ = read_off(off_filename)
    min_pos = points.min(axis=0)
    max_pos = points.max(axis=0)
    np.set_printoptions(suppress=True)
    print("points num: ", points.shape[0])
    print("min_pos: ", min_pos)
    print("max_pos: ", max_pos)

    radius = 0.02 * (max_pos-min_pos).max()
    print("radius for ISS: ", radius)
    ISS(points, radius)

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    # pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    # [_, idx, _] = pcd_tree.search_radius_vector_3d(points[0], radius)
    # neighbor_indices = get_neighbor_points(points, 0, radius)
    # idx = np.asarray(idx)
    # idx = np.delete(idx, np.where)
    # print("kdtree:", idx[0:20])
    # indices = np.where(neighbor_indices == True)
    # indices = np.sort(indices[0])
    # print("raw:", indices[0:20])


if __name__ == '__main__':
    main()
