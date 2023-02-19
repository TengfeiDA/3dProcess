import numpy as np
import open3d as o3d


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


def draw_points_cloud(points):
    # input: points, np.array of [n,3]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    render_option = vis.get_render_option()
    render_option.background_color=np.array([0,0,0]) #设置背景为黑色
    render_option.point_size = 2.0 #设置点云显示尺寸，尺寸越大，点显示效果越粗
    render_option.show_coordinate_frame = True #显示坐标系

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1,1,1])
 
    vis.add_geometry(point_cloud) #添加显示的点云对象

    vis.run() #显示窗口，会阻塞当前线程，直到窗口关闭
    vis.destroy_window() #销毁窗口，该函数必须从主线程调用


def main():
    off_filename = "./data/airplane_0627.off"
    points, _ = read_off(off_filename)
    print("points: ", points.shape)
    draw_points_cloud(points)



if __name__ == '__main__':
    main()
