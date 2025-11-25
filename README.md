# SAM2 Crane

SAM2와 LiDAR를 활용한 크레인 마그넷 3D 포즈 추정 및 측정 시스템

## 프로젝트 개요

크레인 마그넷의 3D 위치/자세를 추정하고 철판과의 거리를 자동 측정하는 시스템입니다.

## 코드별 기능

### 🚀 prediction4.py (최신 버전)
**다중 스케줄 자동 처리** - 여러 스케줄 ID를 자동으로 일괄 처리
- **자동 스케줄 검색**: 지정 디렉토리의 모든 유효한 스케줄 ID 자동 탐색
- **동적 프롬프트 생성**: 각 스케줄마다 DZ 값에 따라 Obj1 프롬프트 자동 생성
- **ML 기반 예측**: ML 모델로 Obj2 프롬프트 자동 예측 (폴백: 기본값)
- **4개 측정값 출력**: P1-P2(상단 길이), P3-P4(하단 길이), P5-P6(상단 너비), P7-P8(하단 너비)
- **에러 복원력**: 한 스케줄 실패 시에도 다음 스케줄 계속 처리
- **결과 요약**: 전체 처리 통계 및 성공/실패 개수 출력

**사용법**:
```bash
python prediction4.py
# → /workspace/sequences_sample/20251123/ 내 모든 스케줄 자동 처리
```

### 📹 prediction2-2.py
**단일 스케줄 처리** - SAM2 비디오 세그멘테이션 + 3D 포즈 추정 + 거리 측정
- SAM2로 비디오에서 마그넷/철판 자동 분할
- 카메라-LiDAR 융합으로 6-DoF 포즈 추정
- ICP로 포즈 정제 (Roll/Pitch 조정)
- 마그넷-철판 간 길이/너비 측정 (mm)
- 결과를 이미지와 CSV로 저장
- **수동 설정**: Config 클래스에서 스케줄 ID 직접 지정 필요

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
# 🚀 다중 스케줄 자동 처리 (추천)
python prediction4.py

# 단일 스케줄 처리 (Config 수정 필요)
python prediction2-2.py

# 포인트 클라우드 확인
python pcd_viz.py

# 거리 측정
python calc_distance_o3d.py
```

## 디렉토리 구조 요구사항

각 스케줄 ID는 다음 구조를 가져야 합니다:

```
sequences_sample/20251123/
├── 202511230054535706/
│   ├── image/      # 이미지 파일 (.jpg, .jpeg)
│   ├── pcd/        # 포인트 클라우드 파일 (.pcd)
│   └── results/    # 측정 결과 (.txt, DZ 값 포함)
├── 202511230141536263/
│   ├── image/
│   ├── pcd/
│   └── results/
└── ...
```

**prediction4.py**는 이 구조를 자동으로 탐색하여 유효한 스케줄만 처리합니다.

## 출력 파일

### 각 스케줄별 출력
- `frame_out_full_vis/{schedule_id}/frame_XXXXX.jpg` - 시각화 이미지 (측정 화살표 포함)
- `frame_out_full_vis/{schedule_id}/measurements.csv` - 측정 데이터 (4개 거리 + 평균)

### CSV 포맷
```csv
frame,P1-P2,P3-P4,P5-P6,P7-P8
0,1234.5,1230.2,450.3,448.7
1,1235.1,1229.8,449.9,449.2
...
AVERAGE,1234.8,1230.0,450.1,449.0
```

- **P1-P2**: 상단 길이 측정 (노란색 화살표)
- **P3-P4**: 하단 길이 측정 (시안 화살표)
- **P5-P6**: 상단 너비 측정 (마젠타 화살표)
- **P7-P8**: 하단 너비 측정 (라임 화살표)
- **AVERAGE**: 첫 프레임 제외 평균값

## 주요 기능

### 자동 프롬프트 생성
- **DZ 기반 Obj1**: 측정된 DZ 거리에 따라 마그넷 프롬프트 포인트 자동 계산
- **ML 기반 Obj2**: 머신러닝 모델(`annotation/model_prompt.pkl`)로 철판 프롬프트 예측
- **Negative 프롬프트**: Obj1 포인트를 Obj2의 negative로 자동 추가하여 분할 정확도 향상

### ICP 정렬
- **제약 조건**: Roll(좌우 기울기) + Pitch(앞뒤 기울기) + Z축 이동만 허용 (Yaw 고정)
- **중심점 보정**: 회전 시 발생하는 Lever Arm Effect 자동 보정
- **평면 생성**: Obj2 마스크로부터 합성 평면 포인트 클라우드 생성하여 ICP 타겟으로 사용

### 다중 측정
- 4개 지점 동시 측정 (상/하단 길이, 상/하단 너비)
- 2D 이미지에 컬러별 화살표로 시각화
- 프레임별 측정값 + 전체 평균 자동 계산

---

**개발**: 2024-2025 | **목적**: 크레인 마그넷 자동 위치 추정
