import numpy as np
import open3d as o3d


def main():
    
    a = np.random.randn(10, 3)
    print(a)
    b = a[0,:] - a
    print(b)
    c = np.linalg.norm(b, axis=1)
    print("\n",c)
    d = c < 1.5
    print("\n",d)
    e = c[d]
    print("\n",e)





if __name__ == '__main__':
    main()
