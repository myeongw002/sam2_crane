"""
SAM2 Video Predictor + Full 3D Visualization
(Pose Estimation + LiDAR Segmentation + Plane Mesh)
"""

import os
import sys
import csv
import copy
import traceback
import re
import glob
import joblib
import numpy as np
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon
from scipy.optimize import least_squares

# ========================================
# 1. 환경 설정
# ========================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# SAM2 경로 설정
SAM2_ROOT = "/workspace/sam2"
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

# ========================================
# 2. 디바이스 설정
# ========================================
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚠️  Using MPS (preliminary support)")
    else:
        device = torch.device("cpu")
        print("ℹ️  Using CPU")
    return device

device = setup_device()

# ========================================
# 3. SAM2 모델 로드
# ========================================
try:
    from sam2.build_sam import build_sam2_video_predictor
    print("✅ SAM2 module imported")
except ImportError as e:
    print(f"❌ SAM2 import failed: {e}")
    sys.exit(1)

sam2_checkpoint = "/workspace/sam2/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# ========================================
# 4. 경로 및 파라미터 설정
# ========================================
class Config:
    ID = "202511250901568327"
    BASE_DIR = f"/workspace/sequences_sample/{ID}"
    VIDEO_DIR = f"{BASE_DIR}/image"
    PCD_DIR = f"{BASE_DIR}/pcd"
    RESULTS_DIR = f"{BASE_DIR}/results"
    INTRINSIC_PATH = "/workspace/sam2/intrinsic.csv"
    EXTRINSIC_PATH = "/workspace/sam2/transform3_tuned_tuned.txt"
    OUTPUT_DIR = f"./frame_out_full_vis/{ID}"
    
    REVERSED = True
    
    # Obj1 프롬프트: DZ 기반 자동 생성 (나중에 업데이트)
    OBJ_1_POINTS = None  # Will be auto-generated from DZ
    OBJ_1_LABELS = np.array([1, 1, 0], dtype=np.int32)  # 3 positive prompts
    
    # Obj2 프롬프트 (고정)
    if not REVERSED :
        OBJ_2_POINTS = np.array([[820, 270], [820, 800]], dtype=np.float32)
        OBJ_2_LABELS = np.array([1, 1], dtype=np.int32)
    else :
        OBJ_2_POINTS = np.array([[830, 490], [830, 670]], dtype=np.float32)
        OBJ_2_LABELS = np.array([1, 1], dtype=np.int32)
    
    
    APPLY_EROSION = True
    EROSION_KERNEL_SIZE = 9
    EROSION_ITERATIONS = 3
    MAX_DEPTH = 15.0
    SHOW_O3D = False

    ACWL_DZ = None  # Will be loaded from results
    DEPTH_TH = None  # Will be calculated from ACWL_DZ

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ========================================
# 5. 프롬프트 예측 함수 (ML Model 기반)
# ========================================
MODEL_PATH = "./annotation/model_prompt.pkl"
N_POINTS_OBJ2 = 4  # OBJ_2 프롬프트 개수 (2~4)

# 로그 파싱 정규식
DZ_PATTERN = re.compile(r"DZ Distance\s*:\s*(\d+)\s*mm")
WIDTH_PATTERN = re.compile(r"Plate Max Width\s*:\s*(\d+)\s*mm")
TOPLEN_PATTERN = re.compile(r"Plate Top Length\s*:\s*(\d+)\s*mm")

def parse_txt_full(path):
    """결과 텍스트 파일에서 DZ, Width, Length 파싱"""
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()
    dz_match = DZ_PATTERN.search(t)
    w_match = WIDTH_PATTERN.search(t)
    L_match = TOPLEN_PATTERN.search(t)
    
    if not dz_match or not w_match or not L_match:
        return None, None, None
    
    dz = int(dz_match.group(1))
    w = int(w_match.group(1))
    L = int(L_match.group(1))
    return dz, w, L

def predict_obj2_prompts(schedule_id, n_points=N_POINTS_OBJ2):
    """
    ML 모델을 사용하여 OBJ_2 프롬프트 자동 예측
    
    Args:
        schedule_id: 스케줄 ID
        n_points: 생성할 프롬프트 개수 (2~4)
    
    Returns:
        prompts: np.ndarray (n_points, 2) 또는 None (실패 시)
    """
    if not (2 <= n_points <= 4):
        print(f"   ⚠️ n_points는 2~4 사이여야 합니다. 현재 값: {n_points}")
        return None
    
    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"   ⚠️ ML 모델 파일 없음: {MODEL_PATH}")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"   ⚠️ 모델 로드 실패: {e}")
        return None
    
    # Results 디렉토리에서 첫 번째 txt 파일 찾기
    base_dir = f"/workspace/sequences_sample/{schedule_id}"
    results_dir = os.path.join(base_dir, "results")
    
    if not os.path.exists(results_dir):
        print(f"   ⚠️ Results 디렉토리 없음: {results_dir}")
        return None
    
    txts = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
    if not txts:
        print(f"   ⚠️ Results txt 파일 없음: {results_dir}")
        return None
    
    txt_path = txts[0]
    dz, w, L = parse_txt_full(txt_path)
    
    if dz is None or w is None or L is None:
        print(f"   ⚠️ DZ/Width/Length 파싱 실패: {txt_path}")
        return None
    
    # ML 모델로 각 포인트 예측
    prompts = []
    for point_idx in range(1, n_points + 1):
        X_in = np.array([[dz, w, L, point_idx]], dtype=float)
        try:
            u_pred, v_pred = model.predict(X_in)[0]
            prompts.append([int(round(u_pred)), int(round(v_pred))])
        except Exception as e:
            print(f"   ⚠️ Point {point_idx} 예측 실패: {e}")
            return None
    
    return np.array(prompts, dtype=np.float32)

# ========================================
# 6. 자동 프롬프트 생성 함수 (DZ 기반 - OBJ_1용)
# ========================================
def get_prompt_points_from_dz(dz: float):
    """
    DZ(mm)에 따라 프롬프트 포인트를 자동 결정.
    - dz = 2346 → [(1020,400), (1020,740), (1020,570)]
    - dz = 673  → [(1010,450), (1010,710), (1010,580)]
    - 범위 밖도 clamp 없이 그대로 선형 외삽
    
    Returns:
        points: np.ndarray (3,2)
    """
    # Anchor values
    dz_far  = 2346
    pts_far = np.array([[1020, 400],
                        [1020, 740],
                        [1020, 570]], dtype=np.float32)

    dz_near = 673
    pts_near = np.array([[1010, 450],
                         [1010, 710],
                         [1010, 580]], dtype=np.float32)

    # t=0 → near, t=1 → far (범위 밖도 그대로 외삽)
    t = (dz - dz_near) / float(dz_far - dz_near)

    # Linear interpolation (extrapolation allowed)
    pts = (1.0 - t) * pts_near + t * pts_far

    return pts.astype(np.float32)

def parse_dz_from_results(results_dir, frame_idx=0):
    """
    Parse DZ Distance value from results text file
    
    Args:
        results_dir: Path to results directory
        frame_idx: Frame index to find corresponding result file
    
    Returns:
        dz_value: DZ Distance in mm, or None if not found
    """
    if not os.path.exists(results_dir):
        return None
    
    # Get all txt files sorted
    txt_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.txt')])
    
    if frame_idx >= len(txt_files):
        return None
    
    txt_path = os.path.join(results_dir, txt_files[frame_idx])
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Search for "DZ Distance" line using regex
        match = re.search(r'DZ Distance\s*:\s*(\d+)\s*mm', content)
        
        if match:
            dz_value = int(match.group(1))
            print(f"   📄 Found DZ Distance: {dz_value}mm")
            return dz_value
        else:
            print(f"   ⚠️ DZ Distance not found in {txt_files[frame_idx]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error reading {txt_path}: {e}")
        return None

# ========================================
# 7. 카메라 파라미터 로드
# ========================================
def load_camera_params(intrinsic_path, extrinsic_path):
    intrinsic = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
    K = np.array([
        [intrinsic[0], intrinsic[1], intrinsic[2]],
        [0.0,          intrinsic[3], intrinsic[4]],
        [0.0,          0.0,          1.0]
    ], dtype=np.float32)
    D = np.array([intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]], dtype=np.float32)
    T_l2c = np.loadtxt(extrinsic_path, delimiter=',').astype(np.float32)
    print(f"✅ Camera parameters loaded")
    return K, D, T_l2c

K_camera, D_dist, T_l2c = load_camera_params(Config.INTRINSIC_PATH, Config.EXTRINSIC_PATH)

# ========================================
# 6. 유틸리티 함수 (마스크 처리)
# ========================================
# ... (show_mask, apply_erosion, to_bool_mask, clamp_inside 함수는 그대로 유지) ...
def show_mask(mask, ax, obj_id=None, random_color=False):
    """마스크 시각화"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def apply_erosion(mask_bool, kernel_size=5, iterations=1):
    """마스크에 erosion 적용하여 경계를 축소"""
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
    dialated = cv2.dilate(eroded, kernel, iterations=iterations)
    return dialated > 0

def to_bool_mask(mask_np):
    """마스크를 boolean 배열로 변환"""
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    return mask_np.astype(bool)

def mask_to_rotated_box(mask_bool):
    """마스크에서 회전 박스 및 상/하/좌/우 좌표 반환"""
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None, None, None, None, None, None, None, None, None, None, None
    
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cxm, cym), (w, h), ang = rect
    
    angle_deg = (ang + 90.0) % 180.0 if w < h else (ang % 180.0)
    corners = cv2.boxPoints(rect).astype(np.float32)
    
    # AABB
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    mid_y = (ymin + ymax) / 2.0
    
    # 좌표 추출
    top_cx = float(cxm); top_y = float(ymin)
    bottom_cx = float(cxm); bottom_y = float(ymax)
    left_x = float(xmin); left_y = float(mid_y)
    right_x = float(xmax); right_y = float(mid_y)
    
    return corners, angle_deg, (xmin, ymin, xmax, ymax), top_cx, top_y, bottom_cx, bottom_y, left_x, left_y, right_x, right_y

def draw_mask_and_rbox(ax, mask_bool, oid, edge_color, H, W, apply_erosion_flag=True, erosion_kernel=5, erosion_iter=1):
    """마스크와 회전 박스 그리기 및 좌표 반환"""
    show_mask(mask_bool, ax, obj_id=oid)
    
    mask_for_box = apply_erosion(mask_bool, kernel_size=erosion_kernel, iterations=erosion_iter) if apply_erosion_flag else mask_bool
    
    corners, angle_deg, aabb, top_cx, top_y, bottom_cx, bottom_y, left_x, left_y, right_x, right_y = mask_to_rotated_box(mask_for_box)
    
    if corners is None:
        return None, None, None, None, None, None, None, None
    
    poly = Polygon(corners, closed=True, fill=False, linewidth=2, edgecolor=edge_color)
    ax.add_patch(poly)
    
    return top_y, top_cx, bottom_y, bottom_cx, left_y, left_x, right_y, right_x



# ========================================
# 7. Pose Estimation Logic (Model & Functions)
# ========================================

# Magnet Model Dimensions (Meters)
MAGNET_WIDTH = 0.45
MAGNET_LENGTH = 2.25
MAGNET_HEIGHT = 0.191

# 3D 모델 포인트 (상단 면)
model_points_2d = np.array([
    [0.0, 0.0],           # Point 0: TL
    [0.0, MAGNET_LENGTH],   # Point 1: BL
    [MAGNET_WIDTH, MAGNET_LENGTH], # Point 2: BR
    [MAGNET_WIDTH, 0.0]     # Point 3: TR
], dtype=np.float32)

model_points_3d_top = np.hstack([model_points_2d, np.zeros((4, 1))])

def affine_matrix(param):
    """2D Affine 변환 행렬 (XY 평면 이동/회전)"""
    tx, ty, theta = param[:3]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    M = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0,      0,      1]
    ], dtype=float)
    return M

def transform_points_3d(param, points_local_3d):
    """로컬 3D 점들을 월드(카메라 오프셋 기준) 좌표로 변환"""
    pose_mat = affine_matrix(param)
    Z_global = param[3]
    
    pts_xy = points_local_3d[:, :2]
    pts_xy_aug = np.hstack([pts_xy, np.ones((pts_xy.shape[0], 1))])
    pts_transformed_xy = (pose_mat @ pts_xy_aug.T).T
    
    pts_transformed = np.zeros_like(points_local_3d)
    pts_transformed[:, 0] = pts_transformed_xy[:, 0]
    pts_transformed[:, 1] = pts_transformed_xy[:, 1]
    pts_transformed[:, 2] = points_local_3d[:, 2] + Z_global 
    
    return pts_transformed

def projection(param, model_points, intrinsic, distortion):
    """파라미터 -> 3D 변환 -> 2D 투영"""
    object_points_world = transform_points_3d(param, model_points)

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.array([[0], [0], [5.0]], dtype=np.float32) # 고정 오프셋 5.0m

    image_points, _ = cv2.projectPoints(
        object_points_world, rvec, tvec, intrinsic, distortion
    )
    return image_points.reshape(-1, 2)

def cost_function(param, model_points, corner_point, intrinsic, distortion):
    predicted = projection(param, model_points, intrinsic, distortion)
    return (corner_point.astype(np.float64) - predicted).ravel()

def order_points_for_model(pts):
    """
    검출된 4개 코너점을 모델 정의 순서(TL, BL, BR, TR)와 일치하도록 정렬
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL (sum min)
    rect[2] = pts[np.argmax(s)] # BR (sum max)
    
    diff = np.diff(pts, axis=1)
    rect[3] = pts[np.argmin(diff)] # TR (diff = y-x. x가 크므로 diff min)
    rect[1] = pts[np.argmax(diff)] # BL (diff = y-x. y가 큼)
    
    return rect



def filter_points_by_mask(points_3d_cam, mask, K, D, W, H, depth_threshold=None, depth_range=0.1, update_mask=False):
    """
    3D 포인트(Camera Frame) 중 2D 마스크 내부에 위치한 포인트만 필터링
    
    Args:
        points_3d_cam: 3D 포인트 (N, 3)
        mask: 2D 마스크 (H, W)
        K, D: 카메라 파라미터
        W, H: 이미지 크기
        depth_threshold: Z값 임계값. 이 값보다 큰 점들은 필터링됨
        update_mask: True이면 threshold 초과 포인트의 마스크 픽셀을 False로 업데이트
    
    Returns:
        filtered_points: 필터링된 3D 포인트
        updated_mask: update_mask=True일 때 업데이트된 마스크, 아니면 None
    """
    if len(points_3d_cam) == 0 or mask is None:
        return (np.array([]), mask.copy() if update_mask else None) if update_mask else np.array([])

    # 1. 3D -> 2D 투영 (왜곡 보정 포함)
    # rvec, tvec는 0 (이미 Camera Frame이므로)
    img_pts, _ = cv2.projectPoints(points_3d_cam, np.zeros(3), np.zeros(3), K, D)
    img_pts = img_pts.squeeze() # (N, 2)

    # 2. 이미지 범위 체크
    u = img_pts[:, 0]
    v = img_pts[:, 1]
    
    # 좌표가 이미지 범위 안인지 확인
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # 3. Depth threshold 체크 (옵션)
    if depth_threshold is not None:
        depth_min = depth_threshold - depth_range
        depth_max = depth_threshold + 0.25
        depth_valid = (points_3d_cam[:, 2] >= depth_min) & (points_3d_cam[:, 2] <= depth_max)
    else:
        depth_valid = np.ones(len(points_3d_cam), dtype=bool)
    
    # 4. 마스크 확인
    # valid_uv와 depth_valid가 True인 인덱스에 대해서만 마스크 값 조회 (정수 변환)
    combined_valid = valid_uv & depth_valid
    u_valid = u[combined_valid].astype(int)
    v_valid = v[combined_valid].astype(int)
    
    # 마스크가 1(True)인 픽셀인지 확인
    in_mask = mask[v_valid, u_valid]
    
    # combined_valid 통과한 애들 중에서도 in_mask인 애들의 원래 인덱스 찾기
    # 1) combined_valid 인덱스 추출
    indices_in_bounds = np.where(combined_valid)[0]
    # 2) 그 중에서 mask 통과한 인덱스
    final_indices = indices_in_bounds[in_mask]
    
    # 5. 마스크 업데이트 (옵션)
    updated_mask = None
    if update_mask:
        updated_mask = mask.copy()
        # Depth threshold 초과 포인트들의 마스크 픽셀을 False로 설정
        if depth_threshold is not None:
            invalid_depth_indices = np.where(valid_uv & ~depth_valid)[0]
            u_invalid = u[invalid_depth_indices].astype(int)
            v_invalid = v[invalid_depth_indices].astype(int)
            # 범위 체크 후 마스크 업데이트
            valid_coords = (u_invalid >= 0) & (u_invalid < W) & (v_invalid >= 0) & (v_invalid < H)
            u_invalid = u_invalid[valid_coords]
            v_invalid = v_invalid[valid_coords]
            updated_mask[v_invalid, u_invalid] = False
    
    if update_mask:
        return points_3d_cam[final_indices], updated_mask
    else:
        return points_3d_cam[final_indices]



def refine_pose_icp_constrained(source_bottom_points, target_plane_points, max_iteration=30):
    """
    [수정됨] Roll(좌우) + Pitch(앞뒤) 회전 및 Z축 이동 허용.
    Yaw(제자리 회전)는 차단.
    회전 시 발생하는 그네 효과(Lever Arm Effect)를 보정하여 중심점 유지.
    """
    # 1. Source의 중심점 계산 (회전의 기준점)
    centroid_source = np.mean(source_bottom_points, axis=0)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_bottom_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_plane_points)
    
    # 법선 계산
    search_radius = 0.1
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=search_radius))
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=search_radius))
    
    # 2. 일반 ICP 수행
    threshold = 0.3
    T_init = np.identity(4)
    
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    T_full = reg.transformation
    
    # 3. 회전 성분 분해 및 Roll, Pitch 추출
    R = T_full[:3, :3]
    t_icp = T_full[:3, 3]
    
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy) # Pitch 값 추출
        # yaw = np.arctan2(R[1,0], R[0,0]) # Yaw는 무시
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        # yaw = 0

    # 4. 제약된 회전 행렬 재구성 (Roll + Pitch)
    # Rx (Roll)
    c_r, s_r = np.cos(roll), np.sin(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, c_r, -s_r],
        [0, s_r, c_r]
    ])
    
    # Ry (Pitch) - 새로 추가됨
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    Ry = np.array([
        [c_p, 0, s_p],
        [0, 1, 0],
        [-s_p, 0, c_p]
    ])
    
    # 결합된 회전 행렬 (R_new = Ry @ Rx)
    # 순서는 미세한 차이가 있지만 보통 Pitch -> Roll 순이나 그 반대나
    # 평면 정렬용 미세 각도에서는 큰 차이 없음. 여기선 Ry @ Rx 적용.
    R_constrained = Ry @ Rx
    
    # 5. [핵심] 중심점 보정 (Centroid Compensation)
    # 목표 중심점: X, Y는 유지(Obj1 원래 위치), Z는 ICP가 제안한 이동량 반영
    target_centroid = centroid_source.copy()
    target_centroid[2] += t_icp[2] 
    
    # 회전만 적용했을 때 중심점이 어디로 튀는지 계산
    rotated_centroid = R_constrained @ centroid_source
    
    # 그 차이만큼을 Translation으로 설정하여 X,Y 위치를 고정
    t_compensated = target_centroid - rotated_centroid
    
    # 최종 변환 행렬 조립
    T_constrained = np.identity(4)
    T_constrained[:3, :3] = R_constrained
    T_constrained[:3, 3] = t_compensated
    
    # 디버깅 출력
    print(f"   ⚖️ Constrained ICP: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°, dZ={t_icp[2]:.3f}m")
    
    return T_constrained

# ========================================
# 9. 3D 시각화 함수 (Open3D)
# ========================================
def get_3d_box_mesh(param, color=[1, 0, 0]):
    """
    추정된 파라미터로 3D 박스 메쉬 (Solid)와 와이어프레임 (LineSet)을 생성하여 반환
    """
    
    top_face = model_points_3d_top 
    bottom_face = top_face.copy()
    bottom_face[:, 2] += MAGNET_HEIGHT 
    local_box_points = np.vstack([top_face, bottom_face])
    
    world_points = transform_points_3d(param, local_box_points)
    camera_points = world_points + np.array([0, 0, 5.0]) # tvec=[0,0,5] 적용
    
    vertices = camera_points
    
    # 1. Solid Mesh 생성
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], # Top Face
        [4, 6, 5], [4, 7, 6], # Bottom Face
        [0, 3, 7], [0, 7, 4], # Side 1
        [3, 2, 6], [3, 6, 7], # Side 2
        [2, 1, 5], [2, 5, 6], # Side 3
        [1, 0, 4], [1, 4, 5]  # Side 4
    ])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    # 2. Wireframe (LineSet) 생성
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color([0, 0, 0]) # 검은색 외곽선
    
    # [수정] Mesh와 Wireframe 두 객체를 반환
    return mesh, wireframe


def show_geometries_with_backface(geoms, title="Viewer"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True   # 뒷면도 렌더링
    vis.run()
    vis.destroy_window()


def two_stage_plane_fit(points_3d, dist_thresh1=0.10, dist_thresh2=0.03, min_inlier_ratio=0.3, min_inlier_abs=30, max_iterations=5):
    """
    Iterative 평면 피팅 (outlier 제거를 반복적으로 수행)
    
    Parameters:
    - points_3d: 입력 3D 점군
    - dist_thresh1: 1차 피팅 거리 임계값
    - dist_thresh2: 2차 이후 피팅 거리 임계값
    - min_inlier_abs: 최소 inlier 개수
    - max_iterations: 최대 반복 횟수
    
    Returns:
    - normal: 평면 법선 벡터
    - d: 평면 방정식 계수 (ax + by + cz + d = 0)
    - centroid: 평면 중심점
    - inlier_mask: 최종 inlier 마스크
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    N = pts.shape[0]
    if N < 3: 
        return None, None, None, None

    # 1차 피팅 (초기 outlier 제거)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    dist = np.abs((pts - centroid) @ normal)
    inlier_mask = dist < dist_thresh1
    
    if inlier_mask.sum() < min_inlier_abs:
        return normal, -float(np.dot(normal, centroid)), centroid, inlier_mask

    # Iterative refinement
    prev_inlier_count = inlier_mask.sum()
    
    for iteration in range(max_iterations):
        pts_inliers = pts[inlier_mask]
        
        if pts_inliers.shape[0] < 3:
            break
        
        # 현재 inlier들로 평면 재피팅
        centroid = pts_inliers.mean(axis=0)
        pts_centered = pts_inliers - centroid
        _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        
        # 전체 점들에 대해 거리 재계산
        dist = np.abs((pts - centroid) @ normal)
        new_inlier_mask = dist < dist_thresh2
        
        if new_inlier_mask.sum() < min_inlier_abs:
            break
        
        # 수렴 체크: inlier 개수가 변하지 않으면 종료
        current_inlier_count = new_inlier_mask.sum()
        if current_inlier_count == prev_inlier_count:
            print(f"   🔄 Plane fitting converged at iteration {iteration + 1}")
            break
        
        # RMS 거리 계산 (디버깅용)
        rms_dist = np.sqrt(np.mean(dist[new_inlier_mask]**2))
        print(f"   🔄 Iteration {iteration + 1}: Inliers={current_inlier_count}/{N}, RMS={rms_dist:.4f}m")
        
        inlier_mask = new_inlier_mask
        prev_inlier_count = current_inlier_count
    
    # 최종 평면 방정식 계수
    d = -float(np.dot(normal, centroid))
    
    return normal, d, centroid, inlier_mask

def build_rbox_clipped_plane(
    rbox_corners, normal, centroid, K, dist_coeffs, color=(0.0, 0.3, 0.5)
):
    """회전 박스 크기만큼 잘린 평면 메쉬 생성 (왜곡 보정 적용)"""
    if rbox_corners is None: return None
    uv_pts = np.asarray(rbox_corners, dtype=np.float32).reshape(-1, 1, 2)
    if uv_pts.shape[0] < 3: return None

    # 왜곡 제거 -> 정규화된 좌표계로 변환
    xy_undistorted = cv2.undistortPoints(uv_pts, K, dist_coeffs).squeeze()
    
    n = np.asarray(normal, dtype=np.float32); n = n / (np.linalg.norm(n) + 1e-12)
    p0 = np.asarray(centroid, dtype=np.float32)

    verts = []
    for x_n, y_n in xy_undistorted:
        d_ray = np.array([x_n, y_n, 1.0], dtype=np.float32)
        d_ray = d_ray / np.linalg.norm(d_ray)
        denom = float(np.dot(n, d_ray))
        
        if abs(denom) < 1e-6: continue
        t = float(np.dot(n, p0) / denom)
        if t <= 0: continue

        P = d_ray * t
        verts.append(P)

    if len(verts) < 3: return None
    verts = np.array(verts, dtype=np.float64); Kp = verts.shape[0]

    triangles = []
    for i in range(1, Kp - 1): triangles.append([0, i, i + 1])
    triangles = np.array(triangles, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    return mesh


def generate_synthetic_plane_cloud(corners_2d, normal, centroid, K, D, density=0.02):
    """
    Obj2의 2D 코너와 평면 파라미터를 이용해 3D 공간 상의 조밀한 평면 점군 생성
    density: 점 간격 (미터 단위)
    """
    if corners_2d is None or normal is None:
        return None

    # 1. 2D 코너를 Undistort (왜곡 보정)
    uv_pts = np.asarray(corners_2d, dtype=np.float32).reshape(-1, 1, 2)
    xy_undistorted = cv2.undistortPoints(uv_pts, K, D).squeeze()

    # 2. 평면 파라미터 준비
    n = np.asarray(normal, dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-12)
    p0 = np.asarray(centroid, dtype=np.float32)

    # 3. 4개의 3D 코너점 계산 (Raycasting)
    corners_3d = []
    for x_n, y_n in xy_undistorted:
        d_ray = np.array([x_n, y_n, 1.0], dtype=np.float32)
        d_ray = d_ray / np.linalg.norm(d_ray) # 단위 벡터
        
        denom = float(np.dot(n, d_ray))
        if abs(denom) < 1e-6: continue # 시선과 평면이 평행함
        
        t = float(np.dot(n, p0) / denom)
        if t <= 0: continue # 평면이 카메라 뒤에 있음
        
        P = d_ray * t
        corners_3d.append(P)
        
    if len(corners_3d) < 3: return None
    corners_3d = np.array(corners_3d)

    # 4. 4개 점 내부를 채우는 그리드 생성 (Bilinear Interpolation or Meshgrid)
    # 간단하게 AABB를 구해서 그리드를 만들고 평면 방정식으로 Z 투영
    min_xyz = corners_3d.min(axis=0)
    max_xyz = corners_3d.max(axis=0)
    
    x_range = np.arange(min_xyz[0], max_xyz[0], density)
    y_range = np.arange(min_xyz[1], max_xyz[1], density)
    
    # 평면 방정식: nx*x + ny*y + nz*z = dot(n, p0) -> z = (dot(n,p0) - nx*x - ny*y) / nz
    d_plane = np.dot(n, p0)
    
    synthetic_points = []
    if abs(n[2]) > 1e-6: # Z축 성분이 있을 때만 (수직 평면 아님)
        xv, yv = np.meshgrid(x_range, y_range)
        xv = xv.flatten()
        yv = yv.flatten()
        zv = (d_plane - n[0]*xv - n[1]*yv) / n[2]
        
        # 생성된 점들이 4각형 안에 있는지 체크할 수도 있지만, ICP용이므로 AABB 전체 사용해도 무방
        synthetic_points = np.vstack([xv, yv, zv]).T
    else:
        # 수직 평면인 경우 예외처리 (여기선 생략하거나 원본 코너만 사용)
        return corners_3d

    return np.array(synthetic_points, dtype=np.float32)

def get_obj1_bottom_cloud(pose_param):
    """
    Obj1의 현재 Pose를 기준으로 바닥면 포인트 클라우드 생성 (Source for ICP)
    """
    # 모델의 바닥면 정의 (Top에서 Z축으로 높이만큼 내림)
    # 좀 더 조밀하게 만들기 위해 grid 생성
    X, Y = np.meshgrid(np.linspace(0, MAGNET_WIDTH, 20), np.linspace(0, MAGNET_LENGTH, 20))
    local_bottom = np.vstack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), MAGNET_HEIGHT)]).T
    
    # World 변환 -> Camera Frame 변환
    P_world = transform_points_3d(pose_param, local_bottom)
    P_camera = P_world + np.array([0, 0, 5.0]) # Offset 적용
    
    return P_camera

def calculate_length_measurements(magnet_pose, slab_corners_3d):
    """
    ICP 정합이 완료된 마그넷 Pose와 철판 코너를 이용하여 지정된 길이 측정
    
    Args:
        magnet_pose (np.array): 마그넷의 6-DoF 파라미터 [tx, ty, tz, rx, ry, rz]
        slab_corners_3d (np.array): 철판(Obj2)의 3D 코너점 4개 (순서: TL, BL, BR, TR)
        
    Returns:
        len_top (float): 위쪽 700mm 지점 측정 거리 (mm)
        len_bot (float): 아래쪽 700mm 지점 측정 거리 (mm)
    """
    # 1. 마그넷 뒷면(Right Edge) 정의 (Local 좌표계)
    # 마그넷 모델: X축(폭)=0.45, Y축(길이)=2.25
    # 그림상 "뒷면"은 긴 변 중 하나임.
    # 마그넷 중심을 기준으로 좌표를 생각하면 더 쉽지만, 현재 모델은 Top-Left(0,0) 기준임.
    # 가정: 마그넷이 세로로 긴 상태(2.25m)라면, 700mm는 양 끝(0.0과 2.25)에서 안으로 들어온 것.
    
    # 마그넷 로컬 좌표 (단위: m)
    # 기준변(Back Face): X=0.45 (오른쪽 변) 이라고 가정 (좌표계 확인 필요)
    # 위쪽 측정점 (P1): X=0.45, Y = 0.7 (700mm)
    # 아래쪽 측정점 (P2): X=0.45, Y = 2.25 - 0.7 (1.55m)
    # 측정 방향 (Normal): X축 양의 방향 (1, 0, 0) -> 그림상 오른쪽 화살표
    
    local_p1 = np.array([MAGNET_WIDTH, 0.7, 0.0])
    local_p2 = np.array([MAGNET_WIDTH, MAGNET_LENGTH - 0.7, 0.0])
    
    # 방향 벡터 (로컬 X축 방향)
    local_direction = np.array([1.0, 0.0, 0.0]) 

    # 2. 월드 좌표로 변환 (Pose 적용)
    # transform_points_3d 함수는 (N,3) 입력을 받음
    p1_world = transform_points_3d(magnet_pose, local_p1.reshape(1, 3)).flatten()
    p2_world = transform_points_3d(magnet_pose, local_p2.reshape(1, 3)).flatten()
    
    # 방향 벡터 회전 (위치 이동은 제외하고 회전만 적용)
    R_mat, _ = cv2.Rodrigues(magnet_pose[3:]) # Rotation Matrix
    dir_world = R_mat @ local_direction
    dir_world = dir_world / np.linalg.norm(dir_world) # 정규화
    
    # Z축 무시하고 2D 평면(XY)에서 계산 (Top View)
    start_pt_1 = p1_world[:2]
    start_pt_2 = p2_world[:2]
    measure_dir = dir_world[:2]
    measure_dir /= np.linalg.norm(measure_dir) # 2D 정규화

    # 3. 철판의 타겟 변(Target Edge) 찾기
    # 철판 코너 4개 중 측정 방향과 교차할 수 있는 '오른쪽 변'을 찾아야 함.
    # 간단히: 철판 중심 기준으로 측정 방향 쪽에 있는 두 점을 이은 선분
    
    slab_center = np.mean(slab_corners_3d[:, :2], axis=0)
    # 각 코너가 중심에서 어느 방향인지 내적 계산
    # (코너 - 중심) dot (측정방향) 값이 가장 큰 두 점이 Target Edge임
    
    dots = np.dot(slab_corners_3d[:, :2] - slab_center, measure_dir)
    target_idx = np.argsort(dots)[-2:] # 가장 큰 값 2개 인덱스
    
    edge_p1 = slab_corners_3d[target_idx[0], :2]
    edge_p2 = slab_corners_3d[target_idx[1], :2]
    
    # 4. 직선 교차점 계산 (Line-Line Intersection)
    def get_distance_to_line(start_pt, direction, line_p1, line_p2):
        # Ray: P = start + t * dir
        # Line: Q = p1 + u * (p2 - p1)
        # 교차점 찾기 (2D)
        x1, y1 = start_pt
        dx, dy = direction
        x3, y3 = line_p1
        x4, y4 = line_p2
        
        # 평행 검사 (분모)
        denom = dx * (y3 - y4) - dy * (x3 - x4)
        if abs(denom) < 1e-6: return 0.0 # 평행 (교차 안함)
        
        # t 계산 (Ray에서의 거리)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        return t # 미터 단위 거리

    len_top = get_distance_to_line(start_pt_1, measure_dir, edge_p1, edge_p2)
    len_bot = get_distance_to_line(start_pt_2, measure_dir, edge_p1, edge_p2)
    
    # 결과 반환 (mm 단위 변환, 절대값)
    return abs(len_top * 1000), abs(len_bot * 1000), (start_pt_1, start_pt_2, measure_dir)

def calculate_width_measurement(magnet_pose, slab_corners_3d):
    """
    마그넷 위쪽 안쪽 코너에서 철판 윗변까지의 수직 거리를 측정 (부호 포함)
    """
    # 1. 마그넷 기준점 (Measure Point) 정의 - 로컬 좌표계
    # 모델 정의: TL(0,0), BL(0, 2.25), BR(0.45, 2.25), TR(0.45, 0)
    # 그림상 '위쪽 안쪽 코너'는 TL(0,0) 또는 TR(0.45, 0) 중 하나임.
    # 그림의 '마그넷 뒷면'이 오른쪽 변(BR-TR)이라면, '안쪽'은 왼쪽 변(TL-BL)임.
    # 따라서 '위쪽 안쪽 코너'는 **TL (0, 0)** 지점으로 추정됨.
    
    local_measure_pt = np.array([0.0, 0.0, 0.0]) 
    
    # 2. 측정 방향 (Normal Vector) 정의 - 로컬 좌표계
    # 마그넷 윗변(Top Edge)에 수직인 방향.
    # 모델상 윗변은 Y=0 라인. 수직인 바깥 방향은 **Y축 음의 방향 (0, -1, 0)**
    # (이미지 좌표계상 위쪽이 Y감소 방향이라면 -1, 아니라면 좌표계 확인 필요)
    
    local_direction = np.array([0.0, -1.0, 0.0]) 

    # 3. 월드 좌표로 변환 (Pose 적용)
    # 점 변환
    p_measure_world = transform_points_3d(magnet_pose, local_measure_pt.reshape(1, 3)).flatten()
    
    # 방향 벡터 회전 (위치 이동 제외, 회전만)
    R_mat, _ = cv2.Rodrigues(magnet_pose[3:]) 
    dir_world = R_mat @ local_direction
    dir_world = dir_world / np.linalg.norm(dir_world) # 정규화
    
    # 2D 평면(XY) 투영
    start_pt = p_measure_world[:2]     # (x, y)
    measure_dir = dir_world[:2]        # (dx, dy)
    measure_dir /= np.linalg.norm(measure_dir)

    # 4. 철판의 타겟 변(Top Edge) 찾기
    # 철판 코너 4개 중 '가장 위쪽'에 있는 두 점을 찾아야 함.
    # (측정 방향과 가장 멀리 있는 점들, 혹은 Y값이 가장 작은 점들)
    # 측정 방향(위쪽)과 내적(Dot Product)이 가장 큰 두 점을 찾음.
    
    slab_center = np.mean(slab_corners_3d[:, :2], axis=0)
    dots = np.dot(slab_corners_3d[:, :2] - slab_center, measure_dir)
    
    # 내적값이 큰 순서대로 정렬 (방향과 일치하는 쪽)
    target_idx = np.argsort(dots)[-2:] 
    
    edge_p1 = slab_corners_3d[target_idx[0], :2]
    edge_p2 = slab_corners_3d[target_idx[1], :2]
    
    # 5. 부호 있는 거리 계산 (Signed Distance)
    # 점(Start_pt)에서 직선(Edge_p1-p2)까지의 거리
    # 공식: distance = ( (x2-x1)(y1-y0) - (x1-x0)(y2-y1) ) / sqrt(...)
    # 혹은 벡터 투영 방식 사용
    
    # 직선의 법선 벡터 (Edge Normal) 구하기
    edge_vec = edge_p2 - edge_p1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-6: return 0.0, (start_pt, start_pt)
    
    # 직선의 한 점(P1)과 측정점(P0) 벡터
    vec_p0_to_line = edge_p1 - start_pt
    
    # 측정 방향(measure_dir)으로의 거리 성분 추출
    # 측정 방향과 평행하지 않을 수 있으므로, '직선까지의 최단 거리'가 아니라
    # '측정 방향 직선'과 '철판 엣지 직선'의 교점까지의 거리를 구해야 함 (Ray Casting)
    
    # Ray: P = S + t * D
    # Line: Q = P1 + u * (P2 - P1)
    # 교점 T 구하기 (이전에 쓴 함수 재사용)
    
    def get_signed_distance_ray(start, direction, p1, p2):
        x1, y1 = start
        dx, dy = direction
        x3, y3 = p1
        x4, y4 = p2
        
        denom = dx * (y3 - y4) - dy * (x3 - x4)
        if abs(denom) < 1e-6: return 0.0 # 평행
        
        # t: start로부터 교점까지의 거리 (방향 포함)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return t

    dist_signed = get_signed_distance_ray(start_pt, measure_dir, edge_p1, edge_p2)
    
    # 단위 변환 (m -> mm)
    return dist_signed * 1000.0, (start_pt, measure_dir)

# ========================================
# 10. Open3D 시각화 통합 함수
# ========================================
def visualize_full_3d(
    pcd_lidar=None, T_l2c=None, mask_obj1=None, mask_obj2=None, 
    K=None, dist_coeffs=None, W=None, H=None, max_depth=9.2, 
    estimated_box=None, estimated_wireframe=None,obj2_rbox_corners=None,
    icp_generated_points=None,
    target_model_points=None,
    obj1_plane_mesh=None
):
    """
    LiDAR, 추정 Box, 평면 Mesh를 모두 통합하여 시각화
    """
    vis_geometries = []
    obj2_pts_cam = None 

    # ------------------------------------------------------------------
    # 1) LiDAR 포인트 클라우드 처리 (색칠)
    # ------------------------------------------------------------------
    if pcd_lidar is not None and T_l2c is not None:
        pts_l = np.asarray(pcd_lidar.points, dtype=np.float32)
        if pts_l.size > 0:
            pts_h = np.hstack([pts_l, np.ones((len(pts_l), 1), dtype=np.float32)])
            pts_cam = (T_l2c @ pts_h.T).T[:, :3]
            Z = pts_cam[:, 2]
            depth_mask = (Z > 1e-6) & (Z < max_depth)
            pts_cam_filtered = pts_cam[depth_mask]
            Nf = pts_cam_filtered.shape[0]
            colors = np.full((Nf, 3), [0.7, 0.7, 0.7], dtype=np.float32)
            
            if all(p is not None for p in [K, dist_coeffs, W, H]) and Nf > 0:
                # filter_points_by_mask 함수를 사용하여 Obj1, Obj2 포인트 추출
                obj1_pts_cam = filter_points_by_mask(pts_cam_filtered, mask_obj1, K, dist_coeffs, W, H) if mask_obj1 is not None else np.array([])
                obj2_pts_cam = filter_points_by_mask(pts_cam_filtered, mask_obj2, K, dist_coeffs, W, H) if mask_obj2 is not None else np.array([])
                
                # 각 포인트가 어느 마스크에 속하는지 색상 할당
                for i, pt in enumerate(pts_cam_filtered):
                    # Obj1에 속하는지 확인
                    if len(obj1_pts_cam) > 0 and np.any(np.all(np.isclose(obj1_pts_cam, pt, atol=1e-6), axis=1)):
                        colors[i] = [1.0, 0.5, 0.0]  # 🟠 Obj1: 주황색
                    # Obj2에 속하는지 확인
                    elif len(obj2_pts_cam) > 0 and np.any(np.all(np.isclose(obj2_pts_cam, pt, atol=1e-6), axis=1)):
                        colors[i] = [0.5, 1.0, 0.0]  # 🟢 Obj2: 연두색

            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(pts_cam_filtered.astype(np.float64))
            pcd_vis.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            vis_geometries.append(pcd_vis)

            # ----------------------------------------------------------
            # 2) Obj2 평면 피팅 및 R-Box 메쉬 생성 (Obj2가 존재할 때)
            # ----------------------------------------------------------
            if (obj2_pts_cam is not None and obj2_pts_cam.shape[0] >= 3 and 
                obj2_rbox_corners is not None):
                
                
                normal, d, centroid, inlier_mask = two_stage_plane_fit(obj2_pts_cam)

                if normal is not None:
                    inlier_pts = obj2_pts_cam[inlier_mask]
                    if inlier_pts.shape[0] >= 3: centroid = inlier_pts.mean(axis=0)
                    
                    plane_mesh = build_rbox_clipped_plane(
                        obj2_rbox_corners, normal, centroid, K=K, dist_coeffs=dist_coeffs, color=(0.0, 0.3, 0.5)
                    )
                    if plane_mesh: vis_geometries.append(plane_mesh)

    # ------------------------------------------------------------------
    # 3) Estimated Box (Mesh) 추가
    # ------------------------------------------------------------------
    if estimated_box is not None:
        vis_geometries.append(estimated_box)
    if estimated_wireframe is not None:
        vis_geometries.append(estimated_wireframe)
    
    # Obj1 평면 메쉬 추가
    if obj1_plane_mesh is not None:
        vis_geometries.append(obj1_plane_mesh)
    
    if icp_generated_points is not None and len(icp_generated_points) > 0:
        pcd_icp = o3d.geometry.PointCloud()
        pcd_icp.points = o3d.utility.Vector3dVector(icp_generated_points.astype(np.float64))
        pcd_icp.paint_uniform_color([1.0, 0.0, 1.0]) # 자홍색
        vis_geometries.append(pcd_icp)
    if target_model_points is not None and len(target_model_points) > 0:
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target_model_points.astype(np.float64))
        pcd_target.paint_uniform_color([0.0, 1.0, 1.0]) # 청록색
        vis_geometries.append(pcd_target)    
    # 4) 좌표축 및 시각화 실행
    vis_geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    
    if vis_geometries:
        show_geometries_with_backface(vis_geometries, title="Full 3D Visualization (Pose + LiDAR)")

TARGET_DPI = 100

def initialize_sam2_and_prompts():
    """DZ 파싱 + Obj1/Obj2 프롬프트 구성 + SAM2 state 초기화 + 파일 목록 반환"""
    print("\n" + "="*50)
    print("Initializing SAM2 inference state...")
    print("="*50)

    print("\n" + "="*50)
    print("Generating automatic prompts from DZ value...")
    print("="*50)

    # DZ 값 파싱
    acwl_dz = parse_dz_from_results(Config.RESULTS_DIR, frame_idx=0)
    if acwl_dz is None:
        print("   ⚠️ DZ not found in results, using default 972mm")
        acwl_dz = 972

    # Config 업데이트
    Config.ACWL_DZ = acwl_dz
    Config.DEPTH_TH = 10 - (acwl_dz * 0.001) + 0.3

    # Obj1 자동 프롬프트
    Config.OBJ_1_POINTS = get_prompt_points_from_dz(acwl_dz)
    print(f"   🎯 Auto-generated OBJ_1 prompts from DZ={acwl_dz}mm:")
    print(f"      Point 1: ({Config.OBJ_1_POINTS[0][0]:.1f}, {Config.OBJ_1_POINTS[0][1]:.1f})")
    print(f"      Point 2: ({Config.OBJ_1_POINTS[1][0]:.1f}, {Config.OBJ_1_POINTS[1][1]:.1f})")
    print(f"      Point 3: ({Config.OBJ_1_POINTS[2][0]:.1f}, {Config.OBJ_1_POINTS[2][1]:.1f})")

    # Obj2 프롬프트 (ML)
    print(f"\n   🤖 Predicting OBJ_2 prompts using ML model...")
    obj2_predicted = predict_obj2_prompts(Config.ID, n_points=N_POINTS_OBJ2)

    if obj2_predicted is not None:
        print(f"   ✅ ML prediction successful ({len(obj2_predicted)} points):")
        for i, pt in enumerate(obj2_predicted, 1):
            print(f"      Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
        obj2_positive_points = obj2_predicted
    else:
        print(f"   ⚠️ ML prediction failed, using default fixed prompts")
        obj2_positive_points = Config.OBJ_2_POINTS

    # Obj1을 Obj2의 negative 프롬프트로 추가
    obj2_negative_points = Config.OBJ_1_POINTS
    Config.OBJ_2_POINTS = np.vstack([obj2_positive_points, obj2_negative_points])
    Config.OBJ_2_LABELS = np.array(
        [1] * len(obj2_positive_points) + [0] * len(obj2_negative_points),
        dtype=np.int32
    )

    print(f"   🎯 Final OBJ_2 prompts (including OBJ_1 as negative):")
    print(f"      Positive: {len([l for l in Config.OBJ_2_LABELS if l == 1])} points")
    print(f"      Negative: {len([l for l in Config.OBJ_2_LABELS if l == 0])} points (includes OBJ_1)")

    # SAM2 state 초기화
    inference_state = predictor.init_state(video_path=Config.VIDEO_DIR)
    predictor.reset_state(inference_state)

    # Obj1, Obj2 프롬프트 추가
    obj_id_1 = 1
    obj_id_2 = 2

    predictor.add_new_points_or_box(
        inference_state=inference_state, frame_idx=0,
        obj_id=obj_id_1,
        points=Config.OBJ_1_POINTS, labels=Config.OBJ_1_LABELS
    )
    predictor.add_new_points_or_box(
        inference_state=inference_state, frame_idx=0,
        obj_id=obj_id_2,
        points=Config.OBJ_2_POINTS, labels=Config.OBJ_2_LABELS
    )

    # 파일 목록
    frame_names = sorted([
        p for p in os.listdir(Config.VIDEO_DIR)
        if p.endswith(('.jpg', '.jpeg'))
    ])
    pcd_files = sorted([
        p for p in os.listdir(Config.PCD_DIR)
        if p.endswith('.pcd')
    ])

    return inference_state, frame_names, pcd_files, obj_id_1, obj_id_2


def build_video_segments(inference_state):
    """SAM2 propagate_in_video 결과를 video_segments 딕셔너리로 구성"""
    video_segments = {}
    for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[f_idx] = {
            oid: to_bool_mask((mask_logits[i] > 0.0).cpu().numpy())
            for i, oid in enumerate(obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)
        }
    return video_segments


def compute_and_draw_measurements(
    ax,
    f_idx,
    final_vertices,
    slab_corners_3d,
    normal_obj2,
    centroid_obj2,
    T_icp_final,
    estimated_param
):
    """
    - 마그넷 박스(final_vertices)와 철판 코너(slab_corners_3d)를 이용해
      P1-P2, P3-P4, P5-P6, P7-P8 거리를 계산하고
      이미지를 기준으로 화살표 및 텍스트를 그린다.
    - 측정 결과 dict를 반환한다.
    """
    # ========== 준비 ==========
    # 슬래브 중심
    slab_center = np.mean(slab_corners_3d[:, :2], axis=0)

    # 최종 박스 꼭짓점 (이름만 정리)
    top_left_corner    = final_vertices[0]  # TL
    bottom_left_corner = final_vertices[1]  # BL

    # 마그넷 길이 방향 param (0~MAGNET_LENGTH)
    t1 = 0.7 / MAGNET_LENGTH
    t2 = (MAGNET_LENGTH - 0.7) / MAGNET_LENGTH

    # 길이 측정 시작점 (3D)
    measure_pt_top = top_left_corner + t1 * (bottom_left_corner - top_left_corner)
    measure_pt_bot = top_left_corner + t2 * (bottom_left_corner - top_left_corner)

    # 회전 행렬 (Yaw + ICP)
    yaw = estimated_param[2]
    R_theta = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_icp = T_icp_final[:3, :3]
    R_total = R_icp @ R_theta

    # 길이 방향: 로컬 x축 → 월드
    dir_world = R_total @ np.array([1.0, 0.0, 0.0])
    measure_dir_2d = dir_world[:2]
    measure_dir_2d /= (np.linalg.norm(measure_dir_2d) + 1e-12)

    # 보조 함수: ray-line 거리
    def ray_line_dist(start_pt, direction, p1, p2):
        x1, y1 = start_pt
        dx, dy = direction
        x3, y3 = p1
        x4, y4 = p2
        denom = dx * (y3 - y4) - dy * (x3 - x4)
        if abs(denom) < 1e-6:
            return 0.0
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return abs(t)

    # ========== 길이 측정 엣지 선택 ==========
    dots = np.dot(slab_corners_3d[:, :2] - slab_center, measure_dir_2d)
    target_idx = np.argsort(dots)[-2:]
    edge_p1 = slab_corners_3d[target_idx[0], :2]
    edge_p2 = slab_corners_3d[target_idx[1], :2]

    # 길이(위/아래) 계산
    len_top_mm = ray_line_dist(measure_pt_top[:2], measure_dir_2d, edge_p1, edge_p2) * 1000.0
    len_bot_mm = ray_line_dist(measure_pt_bot[:2], measure_dir_2d, edge_p1, edge_p2) * 1000.0

    # ========== 너비 측정 (윗변) ==========
    width_dir_world = R_total @ np.array([0.0, -1.0, 0.0])  # 로컬 -y
    width_dir_2d = width_dir_world[:2]
    width_dir_2d /= (np.linalg.norm(width_dir_2d) + 1e-12)

    dots_width = np.dot(slab_corners_3d[:, :2] - slab_center, width_dir_2d)
    target_idx_width = np.argsort(dots_width)[-2:]
    edge_w1 = slab_corners_3d[target_idx_width[0], :2]
    edge_w2 = slab_corners_3d[target_idx_width[1], :2]

    width_dist = ray_line_dist(top_left_corner[:2], width_dir_2d, edge_w1, edge_w2)
    width_mm = width_dist * 1000.0

    # ========== 너비 측정 (아랫변) ==========
    width_bottom_dir_world = R_total @ np.array([0.0, 1.0, 0.0])  # 로컬 +y
    width_bottom_dir_2d = width_bottom_dir_world[:2]
    width_bottom_dir_2d /= (np.linalg.norm(width_bottom_dir_2d) + 1e-12)

    dots_width_bottom = np.dot(slab_corners_3d[:, :2] - slab_center, width_bottom_dir_2d)
    target_idx_width_bottom = np.argsort(dots_width_bottom)[-2:]
    edge_wb1 = slab_corners_3d[target_idx_width_bottom[0], :2]
    edge_wb2 = slab_corners_3d[target_idx_width_bottom[1], :2]

    width_bottom_dist = ray_line_dist(bottom_left_corner[:2], width_bottom_dir_2d, edge_wb1, edge_wb2)
    width_bottom_mm = width_bottom_dist * 1000.0

    print(f"   📏 Length Top: {len_top_mm:.1f}mm, Bottom: {len_bot_mm:.1f}mm")
    print(f"   📐 Width Top: {width_mm:.1f}mm, Bottom: {width_bottom_mm:.1f}mm")

    # ========== 3D → 2D 투영 후 화살표/텍스트 그리기 ==========
    # 끝점 3D 계산
    end_pt_top_3d  = measure_pt_top  + dir_world * (len_top_mm / 1000.0)
    end_pt_bot_3d  = measure_pt_bot  + dir_world * (len_bot_mm / 1000.0)
    end_pt_width_3d        = top_left_corner    + width_dir_world        * (width_mm / 1000.0)
    end_pt_width_bottom_3d = bottom_left_corner + width_bottom_dir_world * (width_bottom_mm / 1000.0)

    pts_to_project = np.array([
        measure_pt_top,          # 0
        measure_pt_bot,          # 1
        top_left_corner,         # 2
        bottom_left_corner,      # 3
        end_pt_top_3d,           # 4
        end_pt_bot_3d,           # 5
        end_pt_width_3d,         # 6
        end_pt_width_bottom_3d   # 7
    ])

    img_measure_pts, _ = cv2.projectPoints(
        pts_to_project, np.zeros(3), np.zeros(3), K_camera, D_dist
    )
    img_measure_pts = img_measure_pts.reshape(-1, 2)

    p_start_top          = img_measure_pts[0]
    p_start_bot          = img_measure_pts[1]
    p_start_width        = img_measure_pts[2]
    p_start_width_bottom = img_measure_pts[3]

    p_end_top          = img_measure_pts[4]
    p_end_bot          = img_measure_pts[5]
    p_end_width        = img_measure_pts[6]
    p_end_width_bottom = img_measure_pts[7]

    # 길이 위쪽 (노란색)
    ax.plot(p_start_top[0], p_start_top[1], 'o', color='yellow', markersize=6, markeredgecolor='black')
    ax.plot(p_end_top[0],   p_end_top[1],   'x', color='yellow', markersize=6, markeredgecolor='black')
    ax.annotate('', xy=p_end_top, xytext=p_start_top,
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2, shrinkA=0, shrinkB=0))
    mid_top = (p_start_top + p_end_top) / 2
    ax.text(
        mid_top[0], mid_top[1] - 15, f'{len_top_mm:.0f}mm',
        color='yellow', fontsize=9, weight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
    )

    # 길이 아래쪽 (시안색)
    ax.plot(p_start_bot[0], p_start_bot[1], 'o', color='cyan', markersize=6, markeredgecolor='black')
    ax.plot(p_end_bot[0],   p_end_bot[1],   'x', color='cyan', markersize=6, markeredgecolor='black')
    ax.annotate('', xy=p_end_bot, xytext=p_start_bot,
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2, shrinkA=0, shrinkB=0))
    mid_bot = (p_start_bot + p_end_bot) / 2
    ax.text(
        mid_bot[0], mid_bot[1] + 20, f'{len_bot_mm:.0f}mm',
        color='cyan', fontsize=9, weight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
    )

    # 너비 위쪽 (마젠타)
    ax.plot(p_start_width[0], p_start_width[1], 'o', color='magenta', markersize=6, markeredgecolor='black')
    ax.plot(p_end_width[0],   p_end_width[1],   'x', color='magenta', markersize=6, markeredgecolor='black')
    ax.annotate('', xy=p_end_width, xytext=p_start_width,
                arrowprops=dict(arrowstyle='->', color='magenta', lw=2, shrinkA=0, shrinkB=0))
    mid_width = (p_start_width + p_end_width) / 2
    ax.text(
        mid_width[0] - 40, mid_width[1], f'{width_mm:.0f}mm',
        color='magenta', fontsize=9, weight='bold', ha='right',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
    )

    # 너비 아래쪽 (라임)
    ax.plot(p_start_width_bottom[0], p_start_width_bottom[1], 'o', color='lime', markersize=6, markeredgecolor='black')
    ax.plot(p_end_width_bottom[0],   p_end_width_bottom[1],   'x', color='lime', markersize=6, markeredgecolor='black')
    ax.annotate('', xy=p_end_width_bottom, xytext=p_start_width_bottom,
                arrowprops=dict(arrowstyle='->', color='lime', lw=2, shrinkA=0, shrinkB=0))
    mid_width_bottom = (p_start_width_bottom + p_end_width_bottom) / 2
    ax.text(
        mid_width_bottom[0] - 40, mid_width_bottom[1], f'{width_bottom_mm:.0f}mm',
        color='lime', fontsize=9, weight='bold', ha='right',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6)
    )

    # CSV용 레코드 반환
    measurement_record = {
        'frame': f_idx,
        'P1-P2': len_top_mm,
        'P3-P4': len_bot_mm,
        'P5-P6': width_mm,
        'P7-P8': width_bottom_mm
    }
    return measurement_record


def process_single_frame(
    f_idx,
    fname,
    pcd_files,
    video_segments,
    obj_id_1,
    obj_id_2,
    last_successful_pose
):
    img_path = os.path.join(Config.VIDEO_DIR, fname)
    pcd_path = os.path.join(Config.PCD_DIR, pcd_files[f_idx]) if f_idx < len(pcd_files) else None

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    fig_w_inch = W / TARGET_DPI
    fig_h_inch = H / TARGET_DPI
    fig = plt.figure(figsize=(fig_w_inch, fig_h_inch))
    ax = plt.gca()
    ax.imshow(img)
    ax.set_title(f"Frame {f_idx} : Matrix Transform Vis")
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.set_axis_off()

    masks = video_segments.get(f_idx, {})
    mask_obj1 = masks.get(obj_id_1, None)
    mask_obj2 = masks.get(obj_id_2, None)

    estimated_param = None
    obj1_corners = None
    obj2_rbox_corners = None

    # 1. 마스크 및 RBox 추출
    for oid in [obj_id_1, obj_id_2]:
        mask = masks.get(oid)
        if mask is not None:
            edge = "orange" if oid == obj_id_1 else "deepskyblue"
            draw_mask_and_rbox(
                ax, mask, oid, edge, H, W,
                Config.APPLY_EROSION, Config.EROSION_KERNEL_SIZE, Config.EROSION_ITERATIONS
            )
            mask_eroded = apply_erosion(mask, Config.EROSION_KERNEL_SIZE, Config.EROSION_ITERATIONS) if Config.APPLY_EROSION else mask
            corners, *_ = mask_to_rotated_box(mask_eroded)

            if corners is not None:
                if oid == obj_id_1:
                    obj1_corners = corners
                elif oid == obj_id_2:
                    obj2_rbox_corners = corners

    final_box_mesh = None
    final_wireframe = None
    final_icp_points = None
    P_target_icp = None
    obj1_plane_mesh = None
    T_icp_final = np.identity(4)

    measurement_record = None

    if obj1_corners is not None:
        ordered_corners = order_points_for_model(obj1_corners)
        initial_param = last_successful_pose.copy()

        try:
            # 2-1. LS Pose
            res = least_squares(
                cost_function, initial_param,
                args=(model_points_3d_top, ordered_corners, K_camera, D_dist),
                loss='soft_l1'
            )
            estimated_param = res.x
            last_successful_pose = estimated_param.copy()

            T_icp_final = np.identity(4)
            synthetic_plane_cloud = None
            obj2_pts_cam = None
            normal_obj2 = None
            centroid_obj2 = None

            # 2-2. Obj2 포인트 + 필터링
            if pcd_path and os.path.exists(pcd_path) and mask_obj2 is not None:
                pcd = o3d.io.read_point_cloud(pcd_path)
                pts_l = np.asarray(pcd.points, dtype=np.float32)
                pts_h = np.hstack([pts_l, np.ones((len(pts_l), 1), dtype=np.float32)])

                P_full_cam = (T_l2c @ pts_h.T).T[:, :3]
                obj2_pts_cam, mask_obj2_updated = filter_points_by_mask(
                    P_full_cam, mask_obj2, K_camera, D_dist, W, H,
                    depth_threshold=Config.DEPTH_TH, update_mask=True
                )
                print(f"   🔍 Obj2 filtered: {len(obj2_pts_cam)} points (depth < {Config.DEPTH_TH}m)")

                if mask_obj2_updated is not None:
                    mask_obj2 = mask_obj2_updated
                    draw_mask_and_rbox(
                        ax, mask_obj2, obj_id_2, "deepskyblue", H, W,
                        Config.APPLY_EROSION, Config.EROSION_KERNEL_SIZE, Config.EROSION_ITERATIONS
                    )
                    mask_eroded_obj2 = apply_erosion(mask_obj2, Config.EROSION_KERNEL_SIZE, Config.EROSION_ITERATIONS) if Config.APPLY_EROSION else mask_obj2
                    corners_updated, *_ = mask_to_rotated_box(mask_eroded_obj2)
                    if corners_updated is not None:
                        obj2_rbox_corners = corners_updated
                        print(f"   🔄 Obj2 rbox updated with filtered mask")

            # 2-3. Obj2 평면 + ICP
            if obj2_rbox_corners is not None and obj2_pts_cam is not None and len(obj2_pts_cam) > 10:
                normal_obj2, _, centroid_obj2, inlier_mask_obj2 = two_stage_plane_fit(obj2_pts_cam)

                if normal_obj2 is not None:
                    inlier_pts_obj2 = obj2_pts_cam[inlier_mask_obj2]
                    if len(inlier_pts_obj2) > 3:
                        centroid_obj2 = inlier_pts_obj2.mean(axis=0)

                    synthetic_plane_cloud = generate_synthetic_plane_cloud(
                        obj2_rbox_corners, normal_obj2, centroid_obj2, K_camera, D_dist
                    )
                    obj1_bottom_cloud = get_obj1_bottom_cloud(estimated_param)

                    if synthetic_plane_cloud is not None and len(synthetic_plane_cloud) > 10:
                        print(f"   🔌 Aligning Obj1 Bottom to Obj2 Plane (Target Pts: {len(synthetic_plane_cloud)})")
                        T_icp_final = refine_pose_icp_constrained(
                            obj1_bottom_cloud, synthetic_plane_cloud, max_iteration=30
                        )
                        print(f"   ✅ ICP Constrained Result:\n{T_icp_final}")
                        delta_t = np.linalg.norm(T_icp_final[:3, 3])
                        if delta_t > 1.0:
                            print(f"   ⚠️ ICP Delta too large ({delta_t:.2f}m). Ignored.")
                            T_icp_final = np.identity(4)

            # 2-4. Box Mesh
            base_mesh, base_wire = get_3d_box_mesh(estimated_param, color=[1, 0, 0])
            base_mesh.transform(T_icp_final)
            base_wire.transform(T_icp_final)
            final_box_mesh = base_mesh
            final_wireframe = base_wire

            if synthetic_plane_cloud is not None:
                final_icp_points = synthetic_plane_cloud

            # 3. 측정/그리기 (분리된 함수 호출)
            if (
                obj2_rbox_corners is not None
                and len(obj2_rbox_corners) == 4
                and normal_obj2 is not None
                and centroid_obj2 is not None
            ):
                # 슬래브 코너 3D 복원
                uv_pts = np.asarray(obj2_rbox_corners, dtype=np.float32).reshape(-1, 1, 2)
                xy_undist = cv2.undistortPoints(uv_pts, K_camera, D_dist).squeeze()

                slab_corners_3d = []
                n = np.asarray(normal_obj2, dtype=np.float32)
                n = n / (np.linalg.norm(n) + 1e-12)
                p0 = np.asarray(centroid_obj2, dtype=np.float32)

                for x_n, y_n in xy_undist:
                    d_ray = np.array([x_n, y_n, 1.0], dtype=np.float32)
                    d_ray = d_ray / np.linalg.norm(d_ray)
                    denom = float(np.dot(n, d_ray))
                    if abs(denom) < 1e-6:
                        continue
                    t = float(np.dot(n, p0) / denom)
                    if t <= 0:
                        continue
                    P = d_ray * t
                    slab_corners_3d.append(P)

                if len(slab_corners_3d) == 4:
                    slab_corners_3d = np.array(slab_corners_3d)
                    measurement_record = compute_and_draw_measurements(
                        ax,
                        f_idx,
                        np.asarray(final_box_mesh.vertices),
                        slab_corners_3d,
                        normal_obj2,
                        centroid_obj2,
                        T_icp_final,
                        estimated_param
                    )

            # 4. 2D 박스 와이어프레임
            if final_box_mesh is not None:
                tx_verts = np.asarray(final_box_mesh.vertices)
                img_pts, _ = cv2.projectPoints(tx_verts, np.zeros(3), np.zeros(3), K_camera, D_dist)
                img_pts = img_pts.reshape(-1, 2).astype(int)
                lines = [
                    [0,1],[1,2],[2,3],[3,0],
                    [4,5],[5,6],[6,7],[7,4],
                    [0,4],[1,5],[2,6],[3,7]
                ]
                for s, e in lines:
                    ax.plot(
                        [img_pts[s, 0], img_pts[e, 0]],
                        [img_pts[s, 1], img_pts[e, 1]],
                        color='red', linewidth=1.5
                    )

            # 5. Pose 텍스트
            if estimated_param is not None:
                tx, ty, yaw, z = estimated_param
                pose_text = (
                    f"6DOF Pose:\n"
                    f"Trans: ({tx:.2f}, {ty:.2f}, {z:.2f})m\n"
                    f"Rot: ({np.degrees(yaw):.1f})°\n"
                )
                if measurement_record is not None:
                    pose_text += (
                        f"Length: T={measurement_record['P1-P2']:.1f}mm "
                        f"B={measurement_record['P3-P4']:.1f}mm\n"
                        f"Width: T={measurement_record['P5-P6']:.1f}mm "
                        f"B={measurement_record['P7-P8']:.1f}mm"
                    )
                ax.text(
                    20, 40, pose_text,
                    color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5)
                )

        except Exception as e:
            print(f"❌ Error frame {f_idx}: {e}")
            traceback.print_exc()

    # 6. 이미지 저장
    out_path = os.path.join(Config.OUTPUT_DIR, f"frame_{f_idx:05d}.jpg")
    plt.savefig(out_path, dpi=TARGET_DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 7. Open3D 시각화
    if Config.SHOW_O3D and pcd_path and os.path.exists(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        visualize_full_3d(
            pcd_lidar=pcd, T_l2c=T_l2c,
            mask_obj1=mask_obj1, mask_obj2=mask_obj2,
            K=K_camera, dist_coeffs=D_dist, W=W, H=H,
            max_depth=Config.MAX_DEPTH,
            estimated_box=final_box_mesh,
            estimated_wireframe=final_wireframe,
            obj2_rbox_corners=obj2_rbox_corners,
            icp_generated_points=final_icp_points,
            target_model_points=P_target_icp,
            obj1_plane_mesh=obj1_plane_mesh
        )

    return last_successful_pose, measurement_record



def process_all_frames(frame_names, pcd_files, video_segments, obj_id_1, obj_id_2):
    """전체 프레임 루프를 돌면서 측정값 리스트를 생성"""
    last_successful_pose = np.array([0.0, 0.0, 0.0, 0.0])
    measurement_records = []

    for f_idx, fname in enumerate(frame_names):
        last_successful_pose, record = process_single_frame(
            f_idx, fname, pcd_files, video_segments,
            obj_id_1, obj_id_2, last_successful_pose
        )
        if record is not None:
            measurement_records.append(record)

    return measurement_records


def save_measurements_csv(measurement_records):
    """측정값 CSV 및 평균행 저장"""
    if len(measurement_records) == 0:
        print("\n⚠️ No measurements recorded.")
        return

    csv_path = os.path.join(Config.OUTPUT_DIR, "measurements.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'P1-P2', 'P3-P4', 'P5-P6', 'P7-P8']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in measurement_records:
            writer.writerow(record)

        if len(measurement_records) > 1:
            records_for_avg = measurement_records[1:]
            avg_len_top = np.mean([r['P1-P2'] for r in records_for_avg])
            avg_len_bot = np.mean([r['P3-P4'] for r in records_for_avg])
            avg_width_top = np.mean([r['P5-P6'] for r in records_for_avg])
            avg_width_bot = np.mean([r['P7-P8'] for r in records_for_avg])

            writer.writerow({
                'frame': 'AVERAGE',
                'P1-P2': f"{avg_len_top:.6f}",
                'P3-P4': f"{avg_len_bot:.6f}",
                'P5-P6': f"{avg_width_top:.6f}",
                'P7-P8': f"{avg_width_bot:.6f}"
            })

    print(f"\n📊 Measurements saved to: {csv_path}")
    print(f"   Total frames: {len(measurement_records)}")
    if len(measurement_records) > 1:
        print(f"   Average (excluding first frame) - Length Top: {avg_len_top:.6f}mm, Bottom: {avg_len_bot:.6f}mm")
        print(f"   Average (excluding first frame) - Width Top: {avg_width_top:.6f}mm, Bottom: {avg_width_bot:.6f}mm")


def main():
    # 1) SAM2 초기화 + 프롬프트 + 파일 목록
    inference_state, frame_names, pcd_files, obj_id_1, obj_id_2 = initialize_sam2_and_prompts()

    # 2) SAM2 전 프레임 propagate
    video_segments = build_video_segments(inference_state)

    # 3) 프레임별 처리 및 측정
    measurement_records = process_all_frames(
        frame_names, pcd_files, video_segments, obj_id_1, obj_id_2
    )

    print("\n✅ Processing Complete.")

    # 4) CSV 저장
    save_measurements_csv(measurement_records)


if __name__ == "__main__":
    main()