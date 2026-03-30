# 3DGS Implementation, from Scratch 

### Overview
본 저장소는 **"3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)** 원본 논문의 핵심 알고리즘을 파이토치(PyTorch)를 이용해 직접 구현한 프로젝트입니다.

공식 코드의 복잡한 구조를 몇 개의 파이썬 코드만으로 구현하는데 초점을 맞췄습니다.

세부 구현 방법 및 코드에 대한 설명은 아래 링크의 블로그에 있습니다.
<br>
[3DGS 구현](https://april2901.tistory.com/category/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%2C%20%EC%97%B0%EA%B5%AC/3DGS%20%EA%B5%AC%ED%98%84)
<br>

## Training Progress Visualization (0 - 30,000 Iters)
무작위 포인트 클라우드에서 정교한 자전거 씬이 형상화되는 과정을 타임랩스로 확인하세요.

[![3DGS Training Process](https://img.youtube.com/vi/VGCpjcYWWpQ/0.jpg)](https://www.youtube.com/watch?v=VGCpjcYWWpQ)
###  Folder Structure & Usage 

> **NOTE:** 본 저장소에는 **핵심 로직을 담은 파이썬(.py) 소스 코드만** 업로드되어 있습니다. 데이터셋과 필수 라이브러리, 그리고 렌더링에 필요한 rasterizer는 별도로 준비해야 합니다.

전체 폴더 구조는 아래와 같습니다.

<br>

<p align="center">
  <img src="https://github.com/april2901/3DGS/blob/main/git.png" alt="Project Folder Structure" width="30%">
  
</p>
output폴더는 학습의 결과(png, ply파일)을 저장하는 용도입니다.<br> clone을 받아도 보이지 않는 것이 맞습니다.<br>
코드를 처음 실행시 output폴더가 없다고 에러가 나오면 빈 폴더를 사진과 같은 위치에 만들어 주세요.
<br>


<br>

### Prerequisites

본 코드를 돌리기 위해서는 다음 두 가지 파일이 필요합니다

#### 1. Mip-NeRF 360 Dataset
학습에 사용하는 데이터셋입니다. 

* [Mip-NeRF 360 Dataset Download Page](https://jonbarron.info/mipnerf360/) 에서 `360_v2` 데이터를 다운로드하세요.
* 다운로드 후, 압축을 풀어 최상위 `3DGS/360_v2` 구조가 되도록 배치하십시오. py파일들과 같은 위치입니다. 위 사진을 참고하세요.

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
2.  `config.py`의 `OBJECT_NAME`변수의 값을 바꿔 학습할 물체를 지정할 수 있습니다. (데이터셋 안의 이름을 사용해야함)
3.  /3DGS 경로에서 train.py실행
4.  1000번 iter 마다 이미지가 저장됩니다.
5.  ply파일은 5000 iter 마다 저장됩니다.

<br>
<hr>
