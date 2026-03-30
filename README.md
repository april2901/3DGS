# 🚴 3D Gaussian Splatting Core Implementation from Scratch 🚴

### 🔬 Overview
본 저장소는 **"3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)** 원본 논문의 핵심 알고리즘을 파이토치(PyTorch)를 이용해 직접 구현한 교육용/연구용 프로젝트입니다.

공식 코드의 복잡한 구조를 탈피하여, **가우시안 초기화(Initial), 밀도 제어(Density Control - Clone/Split/Prune), Adam 옵티마이저 주소 동기화, 그리고 Opacity Reset** 등 핵심 로직을 `.py` 파일 기반으로 선언적으로 구현하는 데 집중했습니다.

<br>

### 🛠️ Key Implemented Features

* **Gaussian Parameter Initialization:** PCD 데이터를 SH 계수(features_dc), Scaling, Rotation(Quaternion)으로 변환 및 VRAM 로딩.
* **Viewspace Positional Gradient Based Densification:** 원본 논문의 $\epsilon_{\alpha} = 0.005$ 문턱값을 준수하는 **Clone** 및 **Split(1.6x scaling down)** 로직 구현.
* **Adaptive Pruning:** 투명도 하한선(`min_opacity`) 및 화면 대 공간 크기(`max_screen_size`, `scene_extent`) 기반 숙청 로직.
* **Dynamic Optimizer Syncing:** 가우시안 개수가 변할 때 파이토치 Adam 옵티마이저의 파라미터 주소 및 모멘텀 상태(`exp_avg`, `exp_avg_sq`)를 강제 동기화하여 학습의 연속성 보장.
* **Periodic Opacity Reset (every 3k iters):** 노이즈(Floaters)를 제거하고 진짜 물체를 솎아내기 위한 Logit 공간에서의 Opacity 초기화 구현.

<br>

### 📂 Folder Structure & Usage 📄 📂

> **NOTE:** 본 저장소에는 **핵심 로직을 담은 파이썬(.py) 소스 코드만** 업로드되어 있습니다. 데이터셋과 필수 라이브러리, 그리고 렌더링에 필요한 rasterizer는 아래 지침에 따라 별도로 준비해야 합니다.

전체 폴더 구조는 아래와 같습니다. (최상위 `project/` 폴더가 `.git`으로 관리됩니다.)

<br>

<p align="center">
  <img src="path/to/your/captured/structure.png" alt="Project Folder Structure" width="60%">
</p>

<br>


<br>

### Prerequisites

본 코드를 돌리기 위해서는 다음 두 가지 파일이 필요합니다

#### 1. Mip-NeRF 360 Dataset
우리가 학습에 사용하는 데이터셋입니다. 

* [Mip-NeRF 360 Dataset Download Page](https://jonbarron.info/mipnerf360/) 에서 `360_v2` 데이터를 다운로드하세요.
* 다운로드 후, 압축을 풀어 최상위 `3DGS/360_v2` 구조가 되도록 배치하십시오. py파일들과 같은 위치입니다.

#### 2. Diff-Gaussian-Rasterization (CUDA)
Inria 연구팀이 고안한 초고속 **C++/CUDA 래스터라이저**를 그대로 가져와 사용합니다. 

* [Inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 저장소를 `recursive` 옵션으로 clone 하십시오.
    ```bash
    git clone --recursive [https://github.com/graphdeco-inria/gaussian-splatting.git](https://github.com/graphdeco-inria/gaussian-splatting.git)
    ```
* 해당 폴더 안의 `submodules/diff-gaussian-rasterization/`에서 파이썬 패키지를 빌드 및 설치해야 합니다. (CUDA Toolkit이 필요합니다.)
    ```bash
    pip install submodules/diff-gaussian-rasterization/
    ```

<br>

### How to Run

1.  데이터셋 다운로드 및 Rasterizer 설치 완료 확인.
2.  최상위 폴더에서 학습 실행:
    ```bash
    python 3DGS/train.py
    ```
3.  1000번 iter 마다 이미지가 저장됩니다.
4.  ply파일은 5000 iter 마다 저장됩니다.

<br>
<hr>
