# SAM2 Crane - 크레인 마그넷 3D 포즈 추정 시스템

SAM2(Segment Anything Model 2)와 LiDAR 센서 융합을 활용한 크레인 마그넷 자동 위치 추정 및 측정 시스템입니다.

## 📋 프로젝트 개요

이 프로젝트는 크레인에 부착된 마그넷의 3D 위치와 자세를 실시간으로 추정하고, 철판과의 거리를 자동으로 측정하는 시스템입니다. SAM2를 이용한 비디오 세그멘테이션과 LiDAR 포인트 클라우드를 결합하여 정밀한 6-DoF(Degree of Freedom) 포즈 추정을 수행합니다.

### 주요 기능

- 🎯 **자동 객체 분할**: SAM2를 활용한 비디오 프레임에서 마그넷 및 철판 자동 세그멘테이션
- 📐 **6-DoF 포즈 추정**: 카메라-LiDAR 융합을 통한 마그넷의 3D 위치 및 회전 추정
- 📏 **자동 거리 측정**: 마그넷과 철판 간의 길이 및 너비 측정 (mm 단위)
- 🔄 **ICP 기반 정합**: Point-to-Plane ICP를 통한 포즈 정제 (Roll/Pitch 제약 조건)
- 📊 **3D 시각화**: Open3D를 활용한 실시간 3D 포인트 클라우드 및 메쉬 시각화
- 📈 **측정 데이터 기록**: CSV 형식으로 프레임별 측정값 저장 및 평균 계산

## 🛠️ 시스템 구성

### 하드웨어 요구사항

- RGB 카메라 (왜곡 보정 파라미터 필요)
- LiDAR 센서
- CUDA 지원 GPU (권장: NVIDIA GPU with 8GB+ VRAM)

### 소프트웨어 요구사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 사용 시)
- SAM2 (Segment Anything Model 2)
- Open3D
- OpenCV
- NumPy, SciPy, Matplotlib

## 📦 설치 방법

### 1. 기본 환경 설정

```bash
# 저장소 클론
git clone https://github.com/myeongw002/sam2_crane.git
cd sam2_crane

# Python 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. 의존성 패키지 설치

```bash
# PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기타 패키지 설치
pip install opencv-python open3d numpy scipy matplotlib pillow
```

### 3. SAM2 설치

```bash
# SAM2 저장소 클론 (별도 디렉토리)
cd /workspace  # 또는 원하는 경로
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
pip install -e .

# SAM2 체크포인트 다운로드
cd checkpoints
./download_ckpts.sh
```

### 4. 카메라 보정 파일 준비

다음 파일들을 준비해야 합니다:

- `intrinsic.csv`: 카메라 내부 파라미터 (fx, fy, cx, cy, k1, k2, p1, p2)
- `transform3_tuned_tuned.txt`: LiDAR-Camera 변환 행렬 (4x4)

## 📖 사용 방법

### 1. 메인 예측 스크립트 (prediction2-2.py)

전체 비디오 시퀀스에 대해 마그넷 포즈 추정 및 측정을 수행합니다.

```python
# Config 클래스에서 설정 변경
class Config:
    ID = "202511201026498863"  # 시퀀스 ID
    VIDEO_DIR = f"/workspace/sequences_sample/{ID}/image"
    PCD_DIR = f"/workspace/sequences_sample/{ID}/pcd"
    
    # 프롬프트 포인트 설정 (첫 프레임에서 클릭한 좌표)
    OBJ_1_POINTS = np.array([[1000, 450], [1000, 700], [1000, 575]], dtype=np.float32)
    OBJ_1_LABELS = np.array([1, 1, 0], dtype=np.int32)
    
    # Depth threshold (미터)
    DEPTH_TH = 8.8
    MAX_DEPTH = 15.0
```

실행:
```bash
python prediction2-2.py
```

출력:
- `./frame_out_full_vis/{ID}/frame_XXXXX.jpg`: 2D 시각화 이미지 (측정값 표시)
- `./frame_out_full_vis/{ID}/measurements.csv`: 측정 데이터 (길이, 너비)
- Open3D 창: 3D 포인트 클라우드 및 메쉬 시각화 (Config.SHOW_O3D=True)

### 2. 포인트 클라우드 시각화 (pcd_viz.py)

PCD 파일을 축 기반 컬러맵으로 시각화합니다.

```python
# pcd_viz.py 스크립트 수정
pcd_paths = [
    '/path/to/your/pointcloud.pcd'
]

visualize_pcds(
    pcd_paths,
    axis="z",           # "x", "y", "z" 선택
    cmap_name="turbo"   # viridis, plasma, jet, turbo 등
)
```

실행:
```bash
python pcd_viz.py
```

### 3. 거리 측정 도구 (calc_distance_o3d.py)

Open3D GUI를 사용하여 포인트 클라우드에서 점 간 거리를 수동으로 측정합니다.

```bash
python calc_distance_o3d.py
```

사용법:
- `Shift + 좌클릭`으로 포인트 선택
- 2개씩 선택하면 거리 자동 계산
- 창을 닫으면 결과 출력

### 4. 코너 검출 (find_corner.py)

LSD(Line Segment Detector) 알고리즘을 사용하여 ROI에서 코너점을 검출합니다.

```bash
python find_corner.py
```

사용법:
- 이미지에서 ROI의 좌상단, 우하단 두 점을 클릭
- 자동으로 4개 코너 검출 및 시각화

## 🔧 주요 파라미터 설정

### prediction2-2.py 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `DEPTH_TH` | Obj2(철판) 필터링 depth 임계값 (m) | 8.8 |
| `MAX_DEPTH` | LiDAR 최대 깊이 (m) | 15.0 |
| `APPLY_EROSION` | 마스크 erosion 적용 여부 | True |
| `EROSION_KERNEL_SIZE` | Erosion 커널 크기 | 9 |
| `EROSION_ITERATIONS` | Erosion 반복 횟수 | 3 |
| `SHOW_O3D` | 3D 시각화 활성화 | True |

### 마그넷 모델 파라미터

```python
MAGNET_WIDTH = 0.45      # 폭 (m)
MAGNET_LENGTH = 2.25     # 길이 (m)
MAGNET_HEIGHT = 0.191    # 높이 (m)
```

## 📊 출력 데이터

### measurements.csv 형식

```csv
frame,length_top_mm,length_bottom_mm,width_top_mm,width_bottom_mm
0,245.3,248.7,120.5,118.2
1,246.1,249.2,119.8,117.9
...
AVERAGE,245.8,248.9,120.1,118.0
```

### 2D 시각화 이미지

각 프레임 이미지에는 다음이 표시됩니다:
- 세그멘테이션 마스크 (Obj1: 주황, Obj2: 하늘색)
- 회전 바운딩 박스 (R-Box)
- 3D 박스 투영 (빨간색)
- 측정 화살표:
  - 노란색: 위쪽 길이 측정
  - 시안: 아래쪽 길이 측정
  - 마젠타: 위쪽 너비 측정
  - 라임: 아래쪽 너비 측정
- 6-DoF 포즈 정보 (Translation, Rotation)

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐
│  Video Frames   │
│  (RGB Images)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  SAM2 Video     │────►│ Segmentation │
│  Predictor      │     │  Masks       │
└─────────────────┘     └──────┬───────┘
                               │
         ┌─────────────────────┴────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│  Corner         │                    │  LiDAR Point    │
│  Detection      │                    │  Cloud Filtering│
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │            ┌──────────────┐          │
         └───────────►│ Pose         │◄─────────┘
                      │ Optimization │
                      │ (Least Sq.)  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ ICP Refinement│
                      │ (Point-Plane) │
                      └──────┬───────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
         ┌─────────────┐          ┌─────────────┐
         │ Measurement │          │ 3D Visualize│
         │ Calculation │          │ (Open3D)    │
         └─────────────┘          └─────────────┘
```

## 🔍 핵심 알고리즘

### 1. 포즈 추정 (Pose Estimation)

- **방법**: Least Squares Optimization
- **파라미터**: `[tx, ty, theta, Z]` (4-DoF → 6-DoF with ICP)
- **손실 함수**: 2D 재투영 오차 최소화

### 2. ICP 정합 (ICP Alignment)

- **알고리즘**: Point-to-Plane ICP
- **제약 조건**: Roll/Pitch 회전만 허용 (Yaw 고정)
- **Centroid Compensation**: X,Y 위치 고정, Z 높이만 조정

### 3. 평면 피팅 (Plane Fitting)

- **방법**: Iterative SVD-based fitting
- **Outlier 제거**: 2단계 거리 임계값 적용
- **수렴 조건**: Inlier 개수 변화 없음

## 📝 TODO

향후 개발 계획은 `todo.txt`를 참고하세요:

1. ACWL 프로토콜 기반 마그넷 자동 프롬프트
2. Depth 기반 마그넷 자동 프롬프트

## 🤝 기여

버그 리포트, 기능 제안, Pull Request는 언제나 환영합니다!

## 📄 라이선스

이 프로젝트는 개인 연구 목적으로 개발되었습니다.

## 🙏 감사의 말

- [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/segment-anything-2) - Meta AI Research
- [Open3D](http://www.open3d.org/) - Point Cloud Processing
- [OpenCV](https://opencv.org/) - Computer Vision Library

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**Last Updated**: 2024-11-21
