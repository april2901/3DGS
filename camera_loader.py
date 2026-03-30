import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import struct


import config
from utils import qvec2rotmat, get_projection_matrix, get_world_view_matrix

class Camera(torch.nn.Module):
    def __init__(self, R, T, foV_x, foV_y, image, image_name):
        super(Camera, self).__init__()
        self.R = R
        self.T = T
        self.foV_x = foV_x
        self.foV_y = foV_y
        self.image_name = image_name
        
        # Ground Truth 이미지 (GPU)
        self.original_image = image.clamp(0.0, 1.0).cuda()
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 3DGS 렌더링용 행렬 계산 (World-to-Camera)
        # 4x4 행렬을 만들고 CUDA가 읽기 좋게 전치(Transpose)해서 저장
        self.world_view_transform = torch.tensor(get_world_view_matrix(R, T)).float().transpose(0, 1).cuda()
        self.projection_matrix = get_projection_matrix(znear=0.01, zfar=100.0, fovX=self.foV_x, fovY=self.foV_y).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform @ self.projection_matrix)
        
        # SH 계산을 위한 카메라 중심 좌표 (월드 기준)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def load_cameras( resolution_scale):
    cameras = []
    
    # cameras.bin
    with open(config.CAMERAS_BIN_PATH, "rb") as f:
        f.read(8) # num_cameras 스킵
        # cam_id, model_id, width, height 스킵
        f.read(24) 
        # f만 가져옴
        focal_length = struct.unpack("<d", f.read(8))[0] 

    # images.bin
    with open(config.IMAGES_BIN_PATH, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]
        
        for _ in tqdm(range(num_reg_images)):
            #헤더에서 정보뽑기
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<I dddd ddd I", f.read(64))
            
            # 파일명
            image_name = ""
            while True:
                char = f.read(1).decode("utf-8")
                if char == "\0": break
                image_name += char
            
            # 포인트 데이터 스킵
            num_points = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points * 24, os.SEEK_CUR) 

            # 리사이즈 관련
            pil_image = Image.open(config.IMAGE_PATH+'/images/'+image_name)
            w, h = pil_image.size
            if resolution_scale != 1:
                w, h = int(w / resolution_scale), int(h / resolution_scale)
                pil_image = pil_image.resize((w, h), Image.LANCZOS)
            
            image_tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            
            # 리사이즈 고려 fov계산
            curr_focal = focal_length / resolution_scale
            fov_x = 2 * np.arctan(w / (2 * curr_focal))
            fov_y = 2 * np.arctan(h / (2 * curr_focal))
            
            
            cameras.append(Camera(
                R=qvec2rotmat(np.array([qw, qx, qy, qz])), 
                T=np.array([tx, ty, tz]), 
                foV_x=fov_x, foV_y=fov_y, 
                image=image_tensor, image_name=image_name
            ))

    return cameras