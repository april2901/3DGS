import torch
import torch.nn as nn
import numpy as np
import os
from plyfile import PlyData, PlyElement
import config
from utils import quat_to_rotmat, RGB2SH

class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 가우시안의 5대 파라미터 정의
        self._xyz = nn.Parameter(torch.empty(0))           # 위치 (N, 3)
        self._features_dc = nn.Parameter(torch.empty(0))    # 색상 (N, 1, 3)
        self._features_rest = nn.Parameter(torch.empty(0))
        self._scaling = nn.Parameter(torch.empty(0))        # 크기 (N, 3) - Log domain
        self._rotation = nn.Parameter(torch.empty(0))       # 회전 (N, 4) - Quaternion
        self._opacity = nn.Parameter(torch.empty(0))        # 불투명도 (N, 1) - Logit domain

        #densification을 위해서
        self.xyz_gradient_accum = torch.empty(0) # 누적된 그래디언트 합
        self.denom = torch.empty(0)              # 가우시안이 화면에 나타난 횟수
        self.max_radii2D = torch.empty(0)       # 회면에 맻힌 최대 반지름

    def create_from_pcd(self, points3d_dict, knn_distances):
        print("가우시안 파라미터 초기화 및 GPU로 로딩 시작")
        
        xyz = np.array([p['xyz'] for p in points3d_dict.values()])
        rgb = np.array([p['rgb'] for p in points3d_dict.values()]) / 255.0
        sh_dc = RGB2SH(rgb)
        
        # 가우시안 좌표, 색 관련
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32).cuda())
        self._features_dc = nn.Parameter(torch.tensor(sh_dc, dtype=torch.float32).unsqueeze(1).cuda())
        f_rest = torch.zeros((xyz.shape[0], 8, 3), dtype=torch.float32, device="cuda")
        self._features_rest = nn.Parameter(f_rest)
        
        # scale
        dist_tensor = torch.tensor(knn_distances, dtype=torch.float32).cuda().unsqueeze(1)
        self._scaling = nn.Parameter(torch.log(dist_tensor.repeat(1, 3)))
        
        # 쿼터니언 초기화
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots)
        
        # 투명도
        self._opacity = nn.Parameter(torch.full((xyz.shape[0], 1), -2.19, device="cuda"))


        #densification을 위해서
        num_points = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
        self.denom = torch.zeros((num_points, 1), device="cuda")
        self.max_radii2D = torch.zeros((num_points), device="cuda")

        print(f"{xyz.shape[0]}개의 가우시안 초기화됨")

    def add_densification_stats(self, viewspace_grad, visibility_filter, radii):

        # 현재 뷰의 gradient계산
        grad_norm = torch.norm(viewspace_grad[visibility_filter, :2], dim=-1, keepdim=True)
        
        # 2. 보이는 가우시안한테만 누적
        self.xyz_gradient_accum[visibility_filter] += grad_norm
        
        # 3. 가우시안들이 화면에 나타난 횟수 1씩 증가
        self.denom[visibility_filter] += 1

        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter], 
            radii[visibility_filter]
        )
    
    def reset_densification_stats(self):
        # 누적된 합과 카운터를 다시 0으로 초기화
        self.xyz_gradient_accum.fill_(0.)
        self.denom.fill_(0.)
        self.max_radii2D.fill_(0.)



    def densify_and_clone(self, grad_threshold, scene_extent):
        # 평균 그래디언트 계산
        grads = self.xyz_gradient_accum / self.denom
        grads[torch.isnan(grads)] = 0
        
        # 평균 그래디언트 > 임계 , AND 크기가 작아야함
        selected_pts_mask = torch.where(grads >= grad_threshold, True, False).squeeze()
        selected_pts_mask &= torch.where(torch.max(self.get_scaling, dim=1).values <= 0.01 * scene_extent, True, False)
        
        # 복제
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 기존 파라미터 리스트 뒤에 붙이기
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_split(self, grad_threshold, scene_extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[torch.isnan(grads)] = 0
        
        # 평균 그래디언트 > 임계, AND 크기가 커야함
        selected_pts_mask = torch.where(grads >= grad_threshold, True, False).squeeze()
        selected_pts_mask &= torch.where(torch.max(self.get_scaling, dim=1).values > 0.01 * scene_extent, True, False)

        #분포 내 위치 2개 뽑기
        stds = self.get_scaling[selected_pts_mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(means, stds) #(0,0,0)중심이고 편차가 std인 분포에서 뽑기
        rots = self.get_rotation[selected_pts_mask].repeat(2, 1)
        
        # 회전 행렬을 적용
        new_xyz = torch.bmm(quat_to_rotmat(rots), samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(2, 1)
        
        # 스케일 감소 
        new_scaling = torch.log(self.get_scaling[selected_pts_mask].repeat(2, 1) / 1.6)
        
        # 3. 나머지 속성들은 그대로 복사
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)

        # 4. 새로운 점들 추가, 부모 점들은 삭제
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        padded_prune_mask = torch.zeros(self._xyz.shape[0], device="cuda", dtype=bool)
        
        # 3. 앞부분(부모들 영역)에만 원래의 selected_pts_mask를 이식합니다.
        # selected_pts_mask의 길이는 딱 부모들이 있던 구간까지만 해당함
        padded_prune_mask[:selected_pts_mask.shape[0]] = selected_pts_mask
        
        # 4. 이제 완벽한 차원(208,761)의 마스크를 넘깁니다.
        self.prune_points(padded_prune_mask)


    def densify_and_prune(self, grad_threshold, min_opacity, scene_extent, max_screen_size):
        # 1. 늘리기 (기존 로직 활용)
        self.densify_and_clone(grad_threshold, scene_extent)
        self.densify_and_split(grad_threshold, scene_extent)

        # 2. [추가] 숙청 마스크 만들기
        # 투명도가 너무 낮은(0.005 미만) 애들
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # 화면에 너무 크게 맺히거나 공간상에서 너무 큰 애들
        if max_screen_size:
            #big_points_vs = self.max_radii2D > max_screen_size
            big_points_vs = torch.zeros_like(self.get_opacity, dtype=torch.bool).squeeze()
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # 3. 숙청 집행 (이미 만들어두신 함수 활용)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def reset_opacity(self):
        opacities_new = torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        eps = 1e-7
        opacities_new = torch.clamp(opacities_new, min=eps, max=1-eps)
        opacities_logit = torch.log(opacities_new / (1 - opacities_new))
        
        # [핵심] .data 수정이 아닌, 파라미터 완전 교체
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_logit, "_opacity")
        self._opacity = optimizable_tensors["_opacity"]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opac, new_scaling, new_rot):

        old_params = [
            self._xyz, self._features_dc, self._features_rest, 
            self._opacity, self._scaling, self._rotation
        ]

        # 기존 + 신규 합치기
        # 각 텐서의 차원을 맞춰서 cat 
        d = {
            "_xyz": torch.cat([self._xyz, new_xyz], dim=0),
            "_features_dc": torch.cat([self._features_dc, new_features_dc], dim=0),
            "_features_rest": torch.cat([self._features_rest, new_features_rest], dim=0),
            "_opacity": torch.cat([self._opacity, new_opac], dim=0),
            "_scaling": torch.cat([self._scaling, new_scaling], dim=0),
            "_rotation": torch.cat([self._rotation, new_rot], dim=0)
        }

        #새 텐서로 교체 
        for name, tensor in d.items():
            setattr(self, name, nn.Parameter(tensor.cuda()))

        # Optimizer 업데이트
        self.update_optimizer_after_densify(new_xyz.shape[0], old_params)

        # 4. 통계 변수도 새 개수(N + M)에 맞춰 확장
        num_new = new_xyz.shape[0]
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_new, 1), device="cuda")], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((num_new, 1), device="cuda")], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(num_new, device="cuda")], dim=0)

    def update_optimizer_after_densify(self, num_new, old_params):

        for i, group in enumerate(self.optimizer.param_groups):
            #xyz찾아 새 파라미터로 교체
            name = group["name"]

            old_p = old_params[i]
            stored_state = self.optimizer.state.get(old_p, None)
            
            new_p = getattr(self, name)
            group['params'][0] = new_p

            if stored_state is not None:
                # Adam 상태 확장 (기존 코드와 동일)
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], torch.zeros((num_new, *stored_state["exp_avg"].shape[1:]), device="cuda")], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], torch.zeros((num_new, *stored_state["exp_avg_sq"].shape[1:]), device="cuda")], dim=0)

                # 옛날 주소 데이터 삭제 후 새 주소에 이식
                del self.optimizer.state[old_p] # 에러 안 남!
                self.optimizer.state[new_p] = stored_state




    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    

    def prune_points(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self.update_optimizer_after_pruning(valid_points_mask)
        
        # 1. [순서 변경] 먼저 옵티마이저를 수술합니다.
        #self.update_optimizer_after_pruning(valid_points_mask)

        # 2. [핵심] 받아온 새 객체들을 그대로 모델 변수에 덮어씌웁니다. (신경망 연결!)
        self._xyz = optimizable_tensors["_xyz"]
        self._features_dc = optimizable_tensors["_features_dc"]
        self._features_rest = optimizable_tensors["_features_rest"]
        self._opacity = optimizable_tensors["_opacity"]
        self._scaling = optimizable_tensors["_scaling"]
        self._rotation = optimizable_tensors["_rotation"]

        # 3. 통계 변수들은 그대로 슬라이싱
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def update_optimizer_after_pruning(self, valid_points_mask):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            old_p = group['params'][0]
            stored_state = self.optimizer.state.get(old_p, None)

            # 1. 데이터 슬라이싱 (여기서 에러가 안 나야 함)
            new_p = nn.Parameter(old_p[valid_points_mask])
            group['params'][0] = new_p

            if stored_state is not None:
                # 2. 모멘텀 데이터들도 똑같이 슬라이싱
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]

                del self.optimizer.state[old_p]
                self.optimizer.state[new_p] = stored_state
            
            optimizable_tensors[group["name"]] = new_p


        return optimizable_tensors

    def save_ply(self, path):
        #ply파일로 저장
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy().reshape(-1, 3) 
        f_rest_learned = self._features_rest.detach().cpu().numpy().reshape(-1, 24)
        f_rest_dummy = np.zeros((xyz.shape[0], 21))
        f_rest = np.concatenate([f_rest_learned, f_rest_dummy], axis=1)

        opacity = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # SIBR 뷰어가 기대하는 62개 속성 정의
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        dtype_full += [(f'f_dc_{i}', 'f4') for i in range(3)]
        dtype_full += [(f'f_rest_{i}', 'f4') for i in range(45)] # 45개 유지 (호환성)
        dtype_full += [('opacity', 'f4')]
        dtype_full += [(f'scale_{i}', 'f4') for i in range(3)]
        dtype_full += [(f'rot_{i}', 'f4') for i in range(4)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scale, rotation], axis=1)
        elements[:] = list(map(tuple, attributes))

        # describe 메서드로 에러 없이 바이너리 저장
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=False).write(path)
        
        #print(f"Binary PLY 저장: {path}")


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling) 

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation) 

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity) 
    
    

if __name__ == "__main__":
    from points3d_bin_parser import Point3DLoader
    import knn


    # 1. 데이터 로드
    loader = Point3DLoader(config.POINTS3D_BIN_PATH)
    points = loader.load()
    initial_scales = knn.compute_initial_scaling(points)

    # 2. 모델 생성 및 초기화
    model = GaussianModel().cuda()
    model.create_from_pcd(points, initial_scales)

    # 3. 뷰어가 읽을 수 있는 경로에 저장
    # 주의: iteration_0 폴더 구조를 미리 맞춰주는 것이 좋습니다.
    output_path = config.OUTPUT_PATH+"/point_cloud/iteration_0/point_cloud.ply"
    model.save_ply(output_path)