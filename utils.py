# utils.py
import torch
import torch.nn.functional as F
import numpy as np


C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def quat_to_rotmat(q):

    norm = torch.nn.functional.normalize(q)
    r, x, y, z = norm[:, 0], norm[:, 1], norm[:, 2], norm[:, 3]

    res = torch.zeros((q.shape[0], 3, 3), device=q.device)

    res[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    res[:, 0, 1] = 2 * (x * y - r * z)
    res[:, 0, 2] = 2 * (x * z + r * y)
    res[:, 1, 0] = 2 * (x * y + r * z)
    res[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    res[:, 1, 2] = 2 * (y * z - r * x)
    res[:, 2, 0] = 2 * (x * z - r * y)
    res[:, 2, 1] = 2 * (y * z + r * x)
    res[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return res

def get_world_view_matrix(R, T):
    rt = np.zeros((4, 4))
    rt[:3, :3] = R
    rt[:3, 3] = T
    rt[3, 3] = 1.0
    return rt

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0 # 3DGS는 +Z 방향을 바라봄

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    
    # [수정된 부분] 3DGS 공식 Z축 깊이 매핑 공식
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    return P



def ssim(img1, img2, window_size=11, size_average=True):
    # 0~1 사이의 텐서
    channel = img1.size(-3)
    window = torch.ones((channel, 1, window_size, window_size), device=img1.device) / (window_size**2)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map




def eval_sh_degree_2(dir, sh_coeffs):
    # (여기에 지난번에 얘기한 SH 수식 추가 가능)
    pass