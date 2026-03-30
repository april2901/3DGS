import numpy as np
from scipy.spatial import KDTree
from points3d_bin_parser import Point3DLoader
import config




def compute_initial_scaling(points_dict):

    xyz = np.array([p['xyz'] for p in points_dict.values()])
    print(f"총 {len(xyz)}개의 점으로 KNN 계산.")


    tree = KDTree(xyz)

    #근처 3개 점 거리 평균
    dists, _ = tree.query(xyz, k=4)
    avg_dists = np.mean(dists[:, 1:], axis=1)


    avg_dists = np.maximum(avg_dists, 1e-7)
    
    print("KNN 계산끝")
    return avg_dists

if __name__ == "__main__":
    path = config.POINTS3D_BIN_PATH
    loader = Point3DLoader(path)
    points = loader.load()
    initial_scales = compute_initial_scaling(points)
