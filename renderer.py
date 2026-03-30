import torch
import math
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    viewpoint_camera: 우리가 만든 Camera 객체
    pc: GaussianModel 객체 (xyz, opacity, scaling, rotation 등 포함)
    pipe: Pipeline 설정 (sh_degree 등)
    bg_color: 배경색 Tensor ([0, 0, 0] 등)
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 #메모리 미리 확보, 일단 0으로 채워 (N,3)만큼
    try:
        screenspace_points.retain_grad() #값 버리지 않는 이유: 나중에 다시 보고 불안정하면 쪼개야하니까
    except:
        pass


    #래스터라이저 엔진이 요구하는 값들 채우기
    tan_fovx = math.tan(viewpoint_camera.foV_x * 0.5)
    tan_fovy = math.tan(viewpoint_camera.foV_y * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width), 
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        bg=bg_color, #배경색
        scale_modifier=scaling_modifier, #scale조정용
        viewmatrix=viewpoint_camera.world_view_transform,      # 카메라 클래스
        projmatrix=viewpoint_camera.full_proj_transform,      # 카메라 클래스
        sh_degree=pipe.sh_degree,
        campos=viewpoint_camera.camera_center,                # 카메라 클래스
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz #가우시안 중심점
    means2D = screenspace_points #2차원 투영된 좌표 + depth  = (N,3)
    opacity = pc.get_opacity


    scales = pc.get_scaling
    rotations = pc.get_rotation



    shs = torch.cat([pc._features_dc, pc._features_rest], dim=1) # 두 개 이어붙임

    #렌더링
    rendered_image, radii , _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=override_color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None #공분산은 그냥 엔진보고 계산하라 함
    )
    #radii는 N차원 텐서임 : 2차원에 투영된 타원을 감싸는 원의 반지름 저장

    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }