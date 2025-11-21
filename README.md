# SAM2 Crane

SAM2와 LiDAR를 활용한 크레인 마그넷 3D 포즈 추정 및 측정 시스템

## 프로젝트 개요

크레인 마그넷의 3D 위치/자세를 추정하고 철판과의 거리를 자동 측정하는 시스템입니다.

## 코드별 기능

### 📹 prediction2-2.py
**메인 실행 파일** - SAM2 비디오 세그멘테이션 + 3D 포즈 추정 + 거리 측정
- SAM2로 비디오에서 마그넷/철판 자동 분할
- 카메라-LiDAR 융합으로 6-DoF 포즈 추정
- ICP로 포즈 정제 (Roll/Pitch 조정)
- 마그넷-철판 간 길이/너비 측정 (mm)
- 결과를 이미지와 CSV로 저장

### 🔍 prediction_debug.py
**디버깅용 간단 버전** - prediction2-2.py의 디버그 버전
- 기본 포즈 추정 기능만 포함
- 측정 기능 없음

### 📊 pcd_viz.py
**포인트 클라우드 시각화** - PCD 파일을 컬러맵으로 시각화
- X/Y/Z 축 기반 색상 매핑
- 여러 PCD 파일 동시 시각화

### 📏 calc_distance_o3d.py
**수동 거리 측정 도구** - 포인트 클라우드에서 점 간 거리 측정
- Open3D GUI로 점 선택 (Shift+클릭)
- 선택한 점들 간 3D 거리 계산

### 🔲 find_corner.py
**코너 검출 도구** - 이미지에서 사각형 코너 자동 검출
- LSD 알고리즘으로 직선 검출
- ROI에서 4개 코너점 자동 추출
- 디버그 시각화 제공

### 🧲 magenet_viz.py
**마그넷 3D 시각화** - 추정된 포즈로 마그넷 3D 모델 생성
- 포즈 파라미터로 3D 박스 메쉬 생성
- LiDAR와 함께 시각화

### 📐 magenet_init/magenet_init.py
**기본 포즈 추정** - Z값 고정, XY 위치만 추정
- Affine 변환 기반 최적화
- 카메라 투영으로 검증

### 📐 magenet_init/magenet_init2.py
**확장 포즈 추정** - Z값까지 추정 (4-DoF)
- magenet_init.py 확장 버전
- tx, ty, theta, Z 모두 추정

## 필수 요구사항

- Python 3.8+
- PyTorch 2.0+ (CUDA 권장)
- SAM2
- Open3D, OpenCV, NumPy, SciPy

## 빠른 시작

```bash
# 메인 실행
python prediction2-2.py

# 포인트 클라우드 확인
python pcd_viz.py

# 거리 측정
python calc_distance_o3d.py
```

## 출력 파일

- `frame_out_full_vis/{ID}/frame_XXXXX.jpg` - 시각화 이미지
- `frame_out_full_vis/{ID}/measurements.csv` - 측정 데이터

---

**개발**: 2024-2025 | **목적**: 크레인 마그넷 자동 위치 추정
