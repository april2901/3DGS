import torch
import torchvision
from renderer import render
from gaussianModel import GaussianModel
import config
from camera_loader import load_cameras
import os


class Pipe:
    def __init__(self):
        self.sh_degree = 2
        self.debug = False

def test():
    # 모델 준비 및 초기화
    from points3d_bin_parser import Point3DLoader
    import knn
    
    loader = Point3DLoader(config.POINTS3D_BIN_PATH)
    points = loader.load()
    initial_scales = knn.compute_initial_scaling(points)

    model = GaussianModel().cuda()
    model.create_from_pcd(points, initial_scales)
    
    # 진짜 카메라들 로드 
    print("camera class 리스트 로딩")
    cameras = load_cameras(
        resolution_scale=2.0
    )
    
    # 첫 번째 카메라 선택
    viewpoint_cam = cameras[0]
    print(f"선택된 카메라: {viewpoint_cam.image_name}")

    pipe = Pipe()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # 렌더링
    print("렌더링 시작")
    render_pkg = render(viewpoint_cam, model, pipe, bg_color)
    image = render_pkg["render"] # (3, H, W) 텐서

    # 이미지 저장
    torchvision.utils.save_image(image, "test_output.png")
    print(" test_output.png 생성.")

    # Radii와 Gradient 확인
    print(f"화면에 보이는 가우시안 개수: {render_pkg['radii'].gt(0).sum().item()}")
    print(f"Screenspace points shape: {render_pkg['viewspace_points'].shape}")

if __name__ == "__main__":
    test()