import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torchvision
import random
from tqdm import tqdm
from renderer import render
from gaussianModel import GaussianModel
from camera_loader import load_cameras
import config


from utils import ssim 

class Pipe:
    def __init__(self):
        self.sh_degree = 2
        self.debug = False

def train():

    from points3d_bin_parser import Point3DLoader
    import knn
    
    if not os.path.exists("output"):
        os.makedirs("output")

    loader = Point3DLoader(config.POINTS3D_BIN_PATH)
    points = loader.load()
    initial_scales = knn.compute_initial_scaling(points)

    model = GaussianModel().cuda()
    model.create_from_pcd(points, initial_scales)
    

    resol_scale = 4.0

    print("camera 클래스 로딩")
    cameras = load_cameras(resolution_scale=resol_scale)
    num_iterations = 30001

    #cam_centers = torch.stack([cam.camera_center for cam in cameras])
    #scene_extent = (cam_centers.max(dim=0).values - cam_centers.min(dim=0).values).norm().item()


    cam_centers = torch.stack([cam.camera_center for cam in cameras])
    avg_cam_center = cam_centers.mean(dim=0)
    distances = torch.norm(cam_centers - avg_cam_center, dim=1)
    scene_extent = distances.max().item() * 1.1
    
    print(f">>> 3DGS 공식 Scene Extent: {scene_extent:.2f}")

    # 초기값과 최종값 모두 scene_extent를 곱해줍니다.
    def get_lr_decay(iteration, max_iters=30000, lr_init=0.00016 * scene_extent, lr_final=0.0000016 * scene_extent):
            if iteration > max_iters: return lr_final
            return lr_init * ((lr_final / lr_init) ** (iteration / max_iters))
    
    

    # 1. 파라미터별 학습률
    lrs = {
        "xyz": 0.00016 * scene_extent,
        "f_dc": 0.0025,
        "f_rest": 0.0025 / 20,
        "opacity": 0.05,
        "scaling": 0.005,
        "rotation": 0.001
    }
    
    params = [
        {'params': [model._xyz], 'lr': lrs["xyz"], "name": "_xyz"},
        {'params': [model._features_dc], 'lr': lrs["f_dc"], "name": "_features_dc"},
        {'params': [model._features_rest], 'lr': lrs["f_rest"], "name": "_features_rest"},
        {'params': [model._opacity], 'lr': lrs["opacity"], "name": "_opacity"},
        {'params': [model._scaling], 'lr': lrs["scaling"], "name": "_scaling"},
        {'params': [model._rotation], 'lr': lrs["rotation"], "name": "_rotation"}
    ]
    optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    model.optimizer = optimizer
    
    progress_bar = tqdm(range(0, num_iterations), desc="Training progress")
    
    pipe = Pipe()
    bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")

    
    
    # (선택) 디버깅용 로그: 이 값이 보통 2.0 ~ 10.0 사이로 나와야 정상입니다.
    print(f">>> 안전하게 계산된 Scene Extent: {scene_extent:.2f}")


    #threshold = 0.0002 / resol_scale
    threshold = 0.0002
    test_cam = cameras[0]

    for iteration in progress_bar:
        if iteration % 1000 == 0:
            pipe.sh_degree = min(iteration // 1000, 2)
                
        current_xyz_lr = get_lr_decay(iteration)


        
        for param_group in optimizer.param_groups:
            if param_group["name"] == "_xyz":
                param_group["lr"] = current_xyz_lr



        viewpoint_cam = random.choice(cameras)
        
        render_pkg = render(viewpoint_cam, model, pipe, bg_color)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"] # (N, 3) 2D 좌표+depth
        visibility_filter = render_pkg["visibility_filter"]     # 화면 내 가우시안 표시
        
        # Loss (L1 + SSIM)
        gt_image = viewpoint_cam.original_image
        
        l1_loss = torch.abs(image - gt_image).mean()
        ssim_loss = 1.0 - ssim(image, gt_image)
        loss = (1.0 - 0.2) * l1_loss + 0.2 * ssim_loss
        
        loss.backward()

        model.add_densification_stats(viewspace_point_tensor.grad, visibility_filter, render_pkg["radii"])


        # densification
        if iteration > 600 and iteration < 15000:
            if iteration % 100 == 0:
                size_threshold = 20 if iteration > 3000 else None
                model.densify_and_prune(threshold, 0.01, scene_extent, size_threshold)

                model.reset_densification_stats()
            
            if iteration % 3000 == 0 or iteration == 3000:
                model.reset_opacity()

        optimizer.step() #실제로 가우시안 값 업데이트
        optimizer.zero_grad(set_to_none=True)

        
        if iteration % 1000 == 0:
            count = model.get_xyz.shape[0]
            progress_bar.write(f"Checkpoint saved at iteration {iteration}, gaussian Num : {count}")
            with torch.no_grad(): # 저장할 때는 그래디언트 계산이 필요 없으니 메모리 절약
                test_render = render(test_cam, model, pipe, bg_color)["render"]
                torchvision.utils.save_image(test_render, f"output1/iter_{iteration}.png")

            if iteration % 5000 == 0:
                model.save_ply(f"output1/points_iter_{iteration}.ply")

        
        if iteration % 10 == 0:
            progress_bar.set_description(f"Training ")
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

if __name__ == "__main__":
    train()