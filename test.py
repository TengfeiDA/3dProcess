import numpy as np
import open3d as o3d
import time


def compare(vectors):
    for i, p in enumerate(vectors):
        dist = np.linalg.norm(vectors - p, axis=1)
        dist[i] = np.finfo(np.float32).max
        sort_index = dist.argsort()
        print(sort_index)
        print(dist[sort_index])
        return sort_index


def delete_same_points(points):
    delete_indices = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        print("\ni", i)
        print("delete_indices", delete_indices)
        if delete_indices[i] == False:
            diff = points - point
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = np.finfo(np.float32).max
            dist[delete_indices] = np.finfo(np.float32).max
            indices = np.where(dist <= 1e-3)
            print("indices", indices[0])
            delete_indices[indices[0]] = True
        print("delete_indices", delete_indices)
    return points[~delete_indices]


# a = np.asarray([[1, 1], [2, 2], [1, 1], [1, 1], [2, 2]])
# print("before:", a)
# a = delete_same_points(a)
# print("after:", a)

a = np.random.randn(10, 3)
print(a)
compare(a)
