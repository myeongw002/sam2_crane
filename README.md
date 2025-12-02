# SAM2 Crane

SAM2와 LiDAR를 활용한 크레인 마그넷 3D 포즈 추정 및 측정 시스템

## 프로젝트 개요

크레인 마그넷의 3D 위치/자세를 추정하고 철판과의 거리를 자동 측정하는 시스템입니다.

## 주요 스크립트

### 🚀 Prediction Scripts (예측 및 측정)

#### prediction4.py ⭐ (최신 권장 버전)
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

#### prediction4-1.py
**prediction4.py의 개선 버전** - 추가 기능 및 최적화가 포함된 버전
- prediction4.py와 유사한 다중 스케줄 처리
- 개선된 에러 처리 및 로깅

#### prediction3.py, prediction3-1.py, prediction3-2.py
**중간 버전** - prediction4.py 이전 버전들
- SAM2 비디오 세그멘테이션 + 3D 포즈 추정
- 점진적 기능 개선 과정의 중간 버전들
- 레거시 참고용

#### prediction2-2.py
**단일 스케줄 처리** - SAM2 비디오 세그멘테이션 + 3D 포즈 추정 + 거리 측정
- SAM2로 비디오에서 마그넷/철판 자동 분할
- 카메라-LiDAR 융합으로 6-DoF 포즈 추정
- ICP로 포즈 정제 (Roll/Pitch 조정)
- 마그넷-철판 간 길이/너비 측정 (mm)
- 결과를 이미지와 CSV로 저장
- **수동 설정**: Config 클래스에서 스케줄 ID 직접 지정 필요

#### prediction_debug.py
**디버깅용 간단 버전** - prediction2-2.py의 디버그 버전
- 기본 포즈 추정 기능만 포함
- 측정 기능 없음

### 🤖 ML 및 프롬프트 자동화

#### gp_train.py
**Gaussian Process 모델 학습** - 오차 예측 모델 훈련
- DZ, W, H 값을 입력으로 오차 예측
- 히트맵 생성 및 시각화
- 학습된 모델을 joblib 파일로 저장

#### train_prompt.py
**프롬프트 예측 모델 학습** - ML 기반 Obj2 프롬프트 자동 예측
- DZ, W, L 값으로 철판 프롬프트 좌표 예측
- annotation/model_prompt.pkl 생성

#### prompt_predictor.py
**프롬프트 예측 실행** - 학습된 모델로 프롬프트 자동 생성
- model_prompt.pkl 모델 로드
- DZ, W, L 값에 따라 2~4개 프롬프트 포인트 예측

#### prompt_automation_test.py
**프롬프트 자동화 테스트** - PCD 변환 및 프롬프트 기반 SAM2 테스트
- PCD Transform 및 acwl_dz 기반 필터링
- 필터링된 PCD 투영 시각화
- 고정 프롬프트로 SAM2 결과 검증

### 📊 데이터 수집 및 분석

#### data_annotater.py
**데이터 어노테이션 도구** - 학습 데이터셋 생성
- 스케줄 ID 디렉토리에서 DZ, Width, Length 추출
- 첫 번째 이미지와 PCD 분석
- annotation/dataset.csv 생성

#### gp_data_collect.py
**GP 데이터 수집** - Gaussian Process 학습용 데이터 수집
- 실험 결과 디렉토리에서 측정값 수집
- GT.csv와 비교하여 오차 계산
- gp_dataset.csv 생성

#### extract_image_dz.py
**DZ별 이미지 추출** - DZ 값에 따라 이미지 분류 및 추출
- results 폴더에서 DZ Distance 파싱
- 첫 번째 이미지 복사 및 DZ 값으로 리네임
- DZ별 이미지 수집용

#### gp_dataset_summary.py
**GP 데이터셋 요약** - 데이터셋 통계 및 시각화
- gp_dataset.csv 분석
- 데이터 분포 및 통계 출력

#### sequence_error_summary.py
**시퀀스 오차 요약** - 측정 결과와 GT 비교
- measurements.csv와 GT.csv 비교
- 오차 통계 계산 및 CSV 생성

### 🛠️ 유틸리티 및 시각화

#### pcd_viz.py
**포인트 클라우드 시각화** - PCD 파일을 컬러맵으로 시각화
- X/Y/Z 축 기반 색상 매핑
- 여러 PCD 파일 동시 시각화

#### calc_distance_o3d.py
**수동 거리 측정 도구** - 포인트 클라우드에서 점 간 거리 측정
- Open3D GUI로 점 선택 (Shift+클릭)
- 선택한 점들 간 3D 거리 계산

#### magenet_viz.py
**마그넷 3D 시각화** - 추정된 포즈로 마그넷 3D 모델 생성
- 포즈 파라미터로 3D 박스 메쉬 생성
- LiDAR와 함께 시각화

#### measurement_maker.py
**측정 결과 집계 도구** - 여러 measurements.csv를 하나의 Excel로 집계
- frame_out_full_vis 디렉토리 탐색
- AVERAGE row 추출
- Excel 요약 파일 생성

#### hand_calibration.py
**수동 캘리브레이션 도구** - 카메라-LiDAR 수동 정렬
- ACWL DZ 기반 깊이 필터링
- PCD를 이미지에 투영하여 정렬 확인

### 📐 Pose Estimation 모듈

#### magenet_init/magenet_init.py
**기본 포즈 추정** - Z값 고정, XY 위치만 추정
- Affine 변환 기반 최적화
- 카메라 투영으로 검증

#### magenet_init/magenet_init2.py
**확장 포즈 추정** - Z값까지 추정 (4-DoF)
- magenet_init.py 확장 버전
- tx, ty, theta, Z 모두 추정

## 필수 요구사항

### 소프트웨어
- Python 3.8+
- PyTorch 2.0+ (CUDA 권장)
- SAM2 ([GitHub](https://github.com/facebookresearch/segment-anything-2))
- Open3D
- OpenCV (cv2)
- NumPy, SciPy
- scikit-learn (Gaussian Process)
- joblib (모델 저장/로드)
- pandas (데이터 분석)
- matplotlib (시각화)
- openpyxl (Excel 출력)

### 데이터 구조
- 카메라 내부 파라미터 (intrinsic.csv)
- 카메라-LiDAR 외부 파라미터 (transform3_tuned_tuned.txt)
- 학습된 ML 모델 (annotation/model_prompt.pkl, models/gp_dz_wh_err.joblib)

## 빠른 시작

```bash
# 🚀 다중 스케줄 자동 처리 (추천)
python prediction4.py

# 단일 스케줄 처리 (Config 수정 필요)
python prediction2-2.py

# 프롬프트 예측 모델 학습
python train_prompt.py

# GP 오차 예측 모델 학습
python gp_train.py

# 포인트 클라우드 확인
python pcd_viz.py

# 수동 거리 측정
python calc_distance_o3d.py

# 측정 결과 집계 (Excel)
python measurement_maker.py /path/to/frame_out_full_vis -o results.xlsx
```

## 디렉토리 구조

### 프로젝트 루트
```
sam2_crane/
├── prediction4.py              # 메인 실행 스크립트 (다중 스케줄 처리)
├── prediction4-1.py            # 개선된 버전
├── prediction3*.py             # 중간 버전들
├── prediction2-2.py            # 단일 스케줄 처리
├── prediction_debug.py         # 디버그 버전
│
├── gp_train.py                 # GP 모델 학습
├── train_prompt.py             # 프롬프트 모델 학습
├── prompt_predictor.py         # 프롬프트 예측
├── prompt_automation_test.py   # 프롬프트 자동화 테스트
│
├── data_annotater.py           # 데이터 어노테이션
├── gp_data_collect.py          # GP 데이터 수집
├── extract_image_dz.py         # DZ별 이미지 추출
├── gp_dataset_summary.py       # 데이터셋 요약
├── sequence_error_summary.py   # 오차 요약
│
├── pcd_viz.py                  # PCD 시각화
├── calc_distance_o3d.py        # 거리 측정 도구
├── magenet_viz.py              # 마그넷 시각화
├── measurement_maker.py        # 측정 집계
├── hand_calibration.py         # 수동 캘리브레이션
│
├── annotation/
│   ├── dataset.csv             # 어노테이션 데이터셋
│   └── model_prompt.pkl        # 학습된 프롬프트 모델
│
├── models/
│   ├── gp_dz_wh_err.joblib     # GP 오차 예측 모델
│   └── validation_scatter.png  # 검증 결과 시각화
│
├── magenet_init/
│   ├── magenet_init.py         # 기본 포즈 추정
│   └── magenet_init2.py        # 확장 포즈 추정
│
├── intrinsic.csv               # 카메라 내부 파라미터
├── transform3_tuned_tuned.txt  # 카메라-LiDAR 외부 파라미터
├── GT.csv                      # Ground Truth 측정값
├── gp_dataset.csv              # GP 학습 데이터
└── gp_dataset_clean.csv        # 정제된 GP 데이터
```

### 데이터 디렉토리 구조 (외부)
각 스케줄 ID는 다음 구조를 가져야 합니다:

```
/workspace/sequences_sample/20251123/
├── 202511230054535706/
│   ├── image/                  # 이미지 파일 (.jpg, .jpeg)
│   ├── pcd/                    # 포인트 클라우드 파일 (.pcd)
│   └── results/                # 측정 결과 (.txt, DZ 값 포함)
├── 202511230141536263/
│   ├── image/
│   ├── pcd/
│   └── results/
└── ...
```

**prediction4.py**는 이 구조를 자동으로 탐색하여 유효한 스케줄만 처리합니다.

## 출력 파일

### Prediction 스크립트 출력
- `frame_out_full_vis/{schedule_id}/frame_XXXXX.jpg` - 시각화 이미지 (측정 화살표 포함)
- `frame_out_full_vis/{schedule_id}/results/measurements.csv` - 측정 데이터 (4개 거리 + 평균)

### CSV 포맷 (measurements.csv)
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

### 데이터 수집 출력
- `annotation/dataset.csv` - 어노테이션 데이터 (DZ, W, L)
- `gp_dataset.csv` - GP 학습용 데이터 (DZ, W, H, Error)
- `sequence_error_summary.csv` - 오차 요약 통계

### 모델 파일
- `annotation/model_prompt.pkl` - 프롬프트 예측 모델
- `models/gp_dz_wh_err.joblib` - GP 오차 예측 모델
- `models/validation_scatter.png` - GP 모델 검증 시각화

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

### 머신러닝 기반 예측
- **Gaussian Process**: DZ, W, H 값으로 측정 오차 예측
- **프롬프트 자동화**: DZ, W, L 값으로 최적 프롬프트 좌표 예측
- **히트맵 생성**: DZ별 오차 분포 시각화

## 워크플로우

### 1. 초기 설정
```bash
# SAM2 설치 및 경로 설정
# /workspace/sam2에 SAM2 설치 필요

# 카메라 파라미터 준비
# - intrinsic.csv (카메라 내부 파라미터)
# - transform3_tuned_tuned.txt (카메라-LiDAR 외부 파라미터)
```

### 2. 데이터 수집 및 학습 (선택적)
```bash
# 어노테이션 데이터 생성
python data_annotater.py

# GP 학습 데이터 수집
python gp_data_collect.py

# 프롬프트 모델 학습
python train_prompt.py

# GP 오차 예측 모델 학습
python gp_train.py
```

### 3. 측정 실행
```bash
# 다중 스케줄 자동 처리 (권장)
python prediction4.py

# 또는 단일 스케줄 처리
python prediction2-2.py
```

### 4. 결과 분석
```bash
# 측정 결과 집계
python measurement_maker.py frame_out_full_vis -o results.xlsx

# 오차 분석
python sequence_error_summary.py

# PCD 시각화
python pcd_viz.py
```

## 트러블슈팅

### CUDA 메모리 부족
- 배치 크기 축소
- SAM2 모델을 경량 버전으로 변경
- 이미지 해상도 축소

### 프롬프트 예측 실패
- `annotation/model_prompt.pkl` 모델 파일 확인
- DZ, W, L 값이 학습 범위 내인지 확인
- 수동 프롬프트로 폴백 (기본값 사용)

### ICP 정렬 실패
- PCD 품질 확인 (노이즈 제거)
- 초기 포즈 추정 개선
- ICP 파라미터 튜닝 (max_iteration, threshold)

### 측정값 부정확
- 카메라-LiDAR 캘리브레이션 재확인
- DZ 값 정확도 검증
- Ground Truth와 비교 분석

## 파일 포맷

### intrinsic.csv
```
fx,s,cx,fy,cy,k1,k2,p1,p2
```

### transform3_tuned_tuned.txt
```
4x4 변환 행렬 (카메라 → LiDAR)
```

### GT.csv
```csv
Schedule_ID,P1-P2,P3-P4,P5-P6,P7-P8
202511230054535706,1234.5,1230.2,450.3,448.7
```

## 성능 최적화

### GPU 활용
- CUDA 사용 권장 (PyTorch CUDA 설치)
- TF32 활성화 (Ampere 이상)
- Mixed Precision (bfloat16)

### 배치 처리
- prediction4.py: 다중 스케줄 일괄 처리
- 에러 복원력으로 중단 없이 계속 실행

### 캐싱
- SAM2 모델 메모리 캐싱
- 반복 사용 시 로딩 시간 단축

---

**개발**: 2024-2025 | **목적**: 크레인 마그넷 자동 위치 추정 및 측정
**기술 스택**: SAM2, PyTorch, Open3D, Gaussian Process
**라이센스**: (프로젝트 라이센스 명시)
