"""
SAM2 Prompt Automation Test
- PCD Transform & acwl_dz-based filtering
- Filtered PCD projection visualization
- Fixed prompt SAM2 results
"""

import os
import sys
import re
import numpy as np
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

# ========================================
# 1. Environment Setup
# ========================================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

SAM2_ROOT = "/workspace/sam2"
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

# ========================================
# 2. Device Setup
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
        print("⚠️  Using MPS")
    else:
        device = torch.device("cpu")
        print("ℹ️  Using CPU")
    return device

device = setup_device()

# ========================================
# 3. SAM2 Model Loading
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
# 4. Configuration
# ========================================
class Config:
    ID = "202511230553538109"
    VIDEO_DIR = f"/workspace/sequences_sample/{ID}/image"
    PCD_DIR = f"/workspace/sequences_sample/{ID}/pcd"
    RESULTS_DIR = f"/workspace/sequences_sample/{ID}/results"
    INTRINSIC_PATH = "/workspace/sam2/intrinsic.csv"
    EXTRINSIC_PATH = "/workspace/sam2/transform3_tuned_tuned.txt"
    OUTPUT_DIR = f"./frame_out_prompt_test/{ID}"
    OUTPUT_SAM2_DIR = f"./frame_out_prompt_test/{ID}/sam2_only"
    OUTPUT_PCD_DIR = f"./frame_out_prompt_test/{ID}/pcd_projection"
    
    # Fixed Prompts
    OBJ_1_POINTS = np.array([[1000, 450], [1000, 700], [1000, 575]], dtype=np.float32)
    OBJ_1_LABELS = np.array([1, 1, 0], dtype=np.int32)
    OBJ_2_POINTS = np.array([[830, 490], [830, 670], [1000, 450], [1000, 700]], dtype=np.float32)
    OBJ_2_LABELS = np.array([1, 1, 0, 0], dtype=np.int32)
    
    # Depth filtering based on acwl_dz (will be updated per frame)
    ACWL_DZ = None  # Will be loaded from results
    DEPTH_TH = None  # Will be calculated per frame
    DEPTH_RANGE = 0.1  # ±10cm

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_SAM2_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_PCD_DIR, exist_ok=True)

# ========================================
# 5. Load Camera Parameters
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
# 6. Utility Functions
# ========================================
def show_mask(mask, ax, obj_id=None, random_color=False):
    """Visualize mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def to_bool_mask(mask_np):
    """Convert mask to boolean array"""
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    return mask_np.astype(bool)

def parse_dz_from_results(results_dir, frame_idx):
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
    
    # Get all txt files sorted by modification time
    txt_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.txt')])
    
    if frame_idx >= len(txt_files):
        return None
    
    txt_path = os.path.join(results_dir, txt_files[frame_idx])
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Search for "DZ Distance" line using regex
        # Pattern: "DZ Distance       : <number> mm"
        match = re.search(r'DZ Distance\s*:\s*(\d+)\s*mm', content)
        
        if match:
            dz_value = int(match.group(1))
            print(f"   📄 Found DZ Distance from {txt_files[frame_idx]}: {dz_value}mm")
            return dz_value
        else:
            print(f"   ⚠️ DZ Distance not found in {txt_files[frame_idx]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error reading {txt_path}: {e}")
        return None

def filter_pcd_by_depth(pcd_lidar, T_l2c, depth_th, depth_range):
    """
    Filter point cloud based on depth threshold (acwl_dz-based)
    
    Args:
        pcd_lidar: Open3D PointCloud in LiDAR frame
        T_l2c: Transformation matrix from LiDAR to Camera
        depth_th: Depth threshold (meters)
        depth_range: Range around threshold (±meters)
    
    Returns:
        pcd_camera: Filtered point cloud in camera frame
        pts_camera: Filtered points as numpy array
    """
    pts_l = np.asarray(pcd_lidar.points, dtype=np.float32)
    if len(pts_l) == 0:
        return None, np.array([])
    
    # Transform to camera frame
    pts_h = np.hstack([pts_l, np.ones((len(pts_l), 1), dtype=np.float32)])
    pts_cam = (T_l2c @ pts_h.T).T[:, :3]
    
    # Depth filtering: depth_th - range <= Z <= depth_th + range
    Z = pts_cam[:, 2]
    depth_min = depth_th - depth_range
    depth_max = depth_th + depth_range
    depth_mask = (Z >= depth_min) & (Z <= depth_max)
    
    pts_filtered = pts_cam[depth_mask]
    
    print(f"   🔍 PCD filtered: {len(pts_filtered)}/{len(pts_l)} points "
          f"(depth: {depth_min:.2f}m - {depth_max:.2f}m)")
    
    # Create Open3D point cloud
    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(pts_filtered.astype(np.float64))
    pcd_camera.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
    
    return pcd_camera, pts_filtered

def project_pcd_to_image(pts_camera, K, D, W, H):
    """
    Project 3D points to 2D image
    
    Args:
        pts_camera: 3D points in camera frame (N, 3)
        K, D: Camera intrinsic parameters
        W, H: Image dimensions
    
    Returns:
        img_pts: 2D image points (N, 2)
        valid_mask: Boolean mask for points inside image
    """
    if len(pts_camera) == 0:
        return np.array([]), np.array([], dtype=bool)
    
    # Project to 2D
    img_pts, _ = cv2.projectPoints(pts_camera, np.zeros(3), np.zeros(3), K, D)
    img_pts = img_pts.squeeze()  # (N, 2)
    
    # Check if points are inside image bounds
    u = img_pts[:, 0]
    v = img_pts[:, 1]
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    return img_pts, valid_mask

def build_density_map(img_pts_valid, H, W):
    """
    Build point density map from projected 2D points
    
    Args:
        img_pts_valid: Valid 2D points (N, 2)
        H, W: Image dimensions
    
    Returns:
        density: Density map (H, W) uint8
    """
    density = np.zeros((H, W), np.uint8)
    for u, v in img_pts_valid.astype(int):
        if 0 <= u < W and 0 <= v < H:
            density[v, u] = min(density[v, u] + 200, 255)  # Accumulate point count

    return density

def extract_plate_mask_from_density(density, debug=True):
    """
    Extract plate region mask from density map using morphology and connected components
    """
    H, W = density.shape

    # 1) Blur
    blurred = cv2.GaussianBlur(density, (11, 11), 0)
    if debug:
        cv2.imshow("DEBUG - 1 Blurred", blurred); cv2.waitKey(0)

    # 2) Threshold
    _, bin_img = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imshow("DEBUG - 2 Binary", bin_img); cv2.waitKey(0)

    # 3) Morphology (noise 제거 + 구멍 메우기)
    kernel = np.ones((9, 9), np.uint8)
    # 좁은 판에서도 너무 많이 지우지 않도록 opening은 약하게 / closing만 강하게
    # opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug:
        cv2.imshow("DEBUG - 3 Closed", closed); cv2.waitKey(0)

    # 4) Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

    candidates = []
    for i in range(1, num_labels):  # 0 = background
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # (a) 너무 작은 영역 제거
        if area < 50:
            continue

        aspect = w / (h + 1e-6)

        # (b) 중심은 여전히 이미지 중앙 근처만 허용
        if not (0.2 * W < cx < 0.8 * W):
            continue

        # (c) 점수 설계: 
        #  - 가로로 길면 가점 (aspect ↑)
        #  - 면적이 크면 가점
        #  - aspect 기준은 너무 빡세게 컷하지 않고, 1.0 이상만 기본 조건
        if aspect < 1.0:
            continue

        score = area * (1.0 + 0.3 * (aspect - 1.0))  # aspect가 1→가중치1, 4→대략 1.9배
        candidates.append((i, score, area, aspect, (x, y, w, h, cx, cy)))

    if not candidates:
        print("   ⚠️ [Plate] no strong candidate → fallback to row-band method")
        return fallback_plate_mask_by_row_band(closed, density, debug=debug)

    # 점수 최대인 라벨 선택
    best = max(candidates, key=lambda x: x[1])
    best_label = best[0]
    _, _, best_area, best_aspect, bbox = best
    x, y, w, h, cx, cy = bbox

    print(f"   ✅ Plate mask extracted: label={best_label}, "
          f"area={best_area}, aspect={best_aspect:.2f}, "
          f"bbox=({x},{y},{w},{h}), center=({cx:.1f},{cy:.1f})")

    plate_mask = (labels == best_label).astype(np.uint8)

    if debug:
        cv2.imshow("DEBUG - FINAL Plate Mask", plate_mask * 255); cv2.waitKey(0); cv2.destroyAllWindows()

    return plate_mask

def fallback_plate_mask_by_row_band(closed, density, debug=False):
    """
    Fallback: use row-sum density to find main horizontal band (plate),
    then restrict to that band on 'closed'.
    """
    H, W = density.shape

    # 행 방향 합 (세로로 누적 → 어느 높이 대역의 포인트가 많은지)
    row_sum = density.sum(axis=1).astype(np.float32)

    # 부드럽게
    row_sum_blur = cv2.GaussianBlur(row_sum.reshape(-1, 1), (31, 1), 0).flatten()

    # 최대값 기준으로 threshold 설정
    max_val = row_sum_blur.max()
    if max_val <= 0:
        print("   ⚠️ [Fallback] row_sum is all zero")
        return np.zeros_like(density, dtype=np.uint8)

    thr = max_val * 0.5  # 상위 50% 이상만 사용 (필요하면 0.3~0.7 조정)
    band_mask_1d = row_sum_blur > thr

    # 연속 구간 중 가장 긴 것 선택
    ys = np.where(band_mask_1d)[0]
    if len(ys) == 0:
        print("   ⚠️ [Fallback] no row band found")
        return np.zeros_like(density, dtype=np.uint8)

    # 연속 구간 찾기
    splits = np.where(np.diff(ys) > 1)[0]
    segments = []
    start = 0
    for s in splits:
        segments.append(ys[start:s+1])
        start = s + 1
    segments.append(ys[start:])

    # 가장 긴 y 구간 선택
    seg = max(segments, key=len)
    y_min, y_max = int(seg[0]), int(seg[-1])

    # 이 y-밴드 안에서 closed를 plate로 사용
    plate_mask = np.zeros_like(density, dtype=np.uint8)
    plate_mask[y_min:y_max+1, :] = closed[y_min:y_max+1, :] > 0

    print(f"   ✅ [Fallback] row-band plate: y=[{y_min}, {y_max}]")

    if debug:
        dbg = (plate_mask * 255).astype(np.uint8)
        cv2.imshow("DEBUG - Fallback Plate Mask", dbg); cv2.waitKey(0); cv2.destroyAllWindows()

    return plate_mask



def pick_prompt_points_from_mask(img_pts_valid, plate_mask, num_points=3):
    """
    Pick SAM2 prompt points from plate mask region
    
    Args:
        img_pts_valid: Valid 2D points (N, 2)
        plate_mask: Binary mask of plate region (H, W)
        num_points: Number of prompt points to pick
    
    Returns:
        points: Prompt points (num_points, 2)
        labels: Prompt labels (num_points,) - all positive (1)
    """
    H, W = plate_mask.shape
    inside_pts = []
    
    for u, v in img_pts_valid.astype(int):
        if 0 <= u < W and 0 <= v < H and plate_mask[v, u] > 0:
            inside_pts.append([u, v])
    
    inside_pts = np.array(inside_pts, dtype=np.float32)
    
    if len(inside_pts) == 0:
        print("   ❌ No points inside plate mask")
        return None, None
    
    print(f"   📍 Found {len(inside_pts)} points inside plate mask")
    
    # Pick center + left/right extremes
    center = inside_pts.mean(axis=0)
    left_idx = np.argmin(inside_pts[:, 0])
    right_idx = np.argmax(inside_pts[:, 0])
    
    left = inside_pts[left_idx]
    right = inside_pts[right_idx]
    
    points = np.stack([center, left, right], axis=0)
    labels = np.array([1, 1, 1], dtype=np.int32)  # All positive prompts
    
    print(f"   🎯 Auto-generated prompts:")
    print(f"      Center: ({center[0]:.1f}, {center[1]:.1f})")
    print(f"      Left:   ({left[0]:.1f}, {left[1]:.1f})")
    print(f"      Right:  ({right[0]:.1f}, {right[1]:.1f})")
    
    return points, labels

# ========================================
# 7. Auto Prompt Generation from Frame 0
# ========================================
print("\n" + "="*50)
print("Generating automatic prompts from Frame 0...")
print("="*50)

# Get frame and PCD file lists
frame_names = sorted([p for p in os.listdir(Config.VIDEO_DIR) if p.endswith(('.jpg', '.jpeg'))])
pcd_files = sorted([p for p in os.listdir(Config.PCD_DIR) if p.endswith('.pcd')])

if len(frame_names) == 0 or len(pcd_files) == 0:
    raise RuntimeError("No frames or PCD files found")

# Load first frame and PCD
img0_path = os.path.join(Config.VIDEO_DIR, frame_names[0])
pcd0_path = os.path.join(Config.PCD_DIR, pcd_files[0])

img0 = Image.open(img0_path).convert("RGB")
W, H = img0.size

# Parse DZ from first frame
acwl_dz_0 = parse_dz_from_results(Config.RESULTS_DIR, 0)
if acwl_dz_0 is None:
    print("   ⚠️ DZ not found in results, using default 972mm")
    acwl_dz_0 = 972

depth_th_0 = 10 - (acwl_dz_0 * 0.001) + 0.3

# Load and filter PCD
pcd0 = o3d.io.read_point_cloud(pcd0_path)
pcd_camera_0, pts_camera_0 = filter_pcd_by_depth(pcd0, T_l2c, depth_th_0, Config.DEPTH_RANGE)

# Project to image
img_pts_0, valid_mask_0 = project_pcd_to_image(pts_camera_0, K_camera, D_dist, W, H)
img_pts_valid_0 = img_pts_0[valid_mask_0]

print(f"   📊 Frame 0: {len(img_pts_valid_0)} valid projected points")

# Build density map and extract plate mask
density_0 = build_density_map(img_pts_valid_0, H, W)
plate_mask_0 = extract_plate_mask_from_density(density_0)

# Generate automatic prompts
auto_points, auto_labels = pick_prompt_points_from_mask(img_pts_valid_0, plate_mask_0)

if auto_points is None:
    raise RuntimeError("❌ Failed to generate automatic prompts")

# ========================================
# 8. SAM2 Initialization with Auto Prompts
# ========================================
print("\n" + "="*50)
print("Initializing SAM2 with automatic prompts...")
print("="*50)

inference_state = predictor.init_state(video_path=Config.VIDEO_DIR)
predictor.reset_state(inference_state)

# Add automatic prompts for plate
obj_id_plate = 1

predictor.add_new_points_or_box(
    inference_state=inference_state, 
    frame_idx=0, 
    obj_id=obj_id_plate, 
    points=auto_points, 
    labels=auto_labels
)

# Propagate masks
video_segments = {}
for f_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[f_idx] = {
        oid: to_bool_mask((mask_logits[i] > 0.0).cpu().numpy())
        for i, oid in enumerate(obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)
    }

print(f"✅ SAM2 propagation complete: {len(video_segments)} frames")

# ========================================
# 9. Main Processing Loop
# ========================================
TARGET_DPI = 100
global_dz_value = None

for f_idx, fname in enumerate(frame_names):
    print(f"\n{'='*50}")
    print(f"Processing Frame {f_idx}/{len(frame_names)} - {fname}")
    print(f"{'='*50}")
    
    img_path = os.path.join(Config.VIDEO_DIR, fname)
    pcd_path = os.path.join(Config.PCD_DIR, pcd_files[f_idx]) if f_idx < len(pcd_files) else None
    
    # Load image
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    
    # Get SAM2 masks
    masks = video_segments.get(f_idx, {})
    mask_plate = masks.get(obj_id_plate, None)
    
    # ========================================
    # Figure 1: SAM2 Results Only
    # ========================================
    fig_w_inch = W / TARGET_DPI
    fig_h_inch = H / TARGET_DPI
    fig1 = plt.figure(figsize=(fig_w_inch, fig_h_inch))
    ax1 = plt.gca()
    ax1.imshow(img)
    ax1.set_title(f"Frame {f_idx} - SAM2 Segmentation (Auto Prompt)")
    ax1.set_xlim([0, W])
    ax1.set_ylim([H, 0])
    ax1.set_axis_off()
    
    # Visualize SAM2 masks
    if mask_plate is not None:
        show_mask(mask_plate, ax1, obj_id=obj_id_plate)
    
    # Draw auto-generated prompts on first frame
    if f_idx == 0 and auto_points is not None:
        for i, (pt, label) in enumerate(zip(auto_points, auto_labels)):
            color = 'lime' if label == 1 else 'red'
            ax1.plot(pt[0], pt[1], 'o', color=color, markersize=8, 
                    markeredgecolor='white', markeredgewidth=2)
            ax1.text(pt[0] + 20, pt[1], f'P{i+1}', color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.7))
    
    # Save SAM2 result
    sam2_output_path = os.path.join(Config.OUTPUT_SAM2_DIR, f"frame_{f_idx:05d}.jpg")
    plt.savefig(sam2_output_path, dpi=TARGET_DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    print(f"   ✅ SAM2 saved: {sam2_output_path}")
    
    # ========================================
    # Figure 2: PCD Projection Only
    # ========================================
    fig2 = plt.figure(figsize=(fig_w_inch, fig_h_inch))
    ax2 = plt.gca()
    ax2.imshow(img)
    ax2.set_title(f"Frame {f_idx} - PCD Projection")
    ax2.set_xlim([0, W])
    ax2.set_ylim([H, 0])
    ax2.set_axis_off()
    
    # Parse DZ Distance from results file
    if global_dz_value is None:
        # First frame → read from file
        first_dz = parse_dz_from_results(Config.RESULTS_DIR, 0)
        if first_dz is None:
            print("   ⚠️ First DZ read failed, fallback to 972mm")
            first_dz = 972

        global_dz_value = first_dz
        print(f"   📌 Global DZ value fixed to {global_dz_value} mm")
    else:
        # Following frames → reuse
        print(f"   ↪ Using global DZ value: {global_dz_value} mm")

    acwl_dz = global_dz_value
    depth_th = 10 - (acwl_dz * 0.001) + 0.3

    # Process PCD
    if pcd_path and os.path.exists(pcd_path):
        print(f"   📂 Loading PCD: {pcd_files[f_idx]}")
        pcd_lidar = o3d.io.read_point_cloud(pcd_path)
        
        # 1. Filter PCD by depth (acwl_dz-based)
        pcd_camera, pts_camera = filter_pcd_by_depth(
            pcd_lidar, T_l2c, depth_th, Config.DEPTH_RANGE
        )
        
        # 2. Project filtered PCD to image
        if len(pts_camera) > 0:
            img_pts, valid_mask = project_pcd_to_image(pts_camera, K_camera, D_dist, W, H)
            img_pts_valid = img_pts[valid_mask]
            
            print(f"   📍 Projected points: {len(img_pts_valid)}/{len(pts_camera)}")
            
            # 3. Visualize projected points
            if len(img_pts_valid) > 0:
                ax2.scatter(img_pts_valid[:, 0], img_pts_valid[:, 1], 
                          c='red', s=3, alpha=0.7, label='Filtered PCD')
        else:
            print("   ⚠️ No points after depth filtering")
    else:
        print(f"   ⚠️ PCD file not found: {pcd_path}")
    
    # Add legend
    ax2.legend(loc='upper right', fontsize=8)
    
    # Add info text
    info_text = (
        f"ACWL_DZ: {acwl_dz}mm\n"
        f"Depth: {depth_th - Config.DEPTH_RANGE:.2f}m - {depth_th + Config.DEPTH_RANGE:.2f}m"
    )
    ax2.text(20, 40, info_text, color='white', fontsize=10, 
           bbox=dict(facecolor='black', alpha=0.7))
    
    # Save PCD projection result
    pcd_output_path = os.path.join(Config.OUTPUT_PCD_DIR, f"frame_{f_idx:05d}.jpg")
    plt.savefig(pcd_output_path, dpi=TARGET_DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    
    print(f"   ✅ PCD saved: {pcd_output_path}")

print("\n" + "="*50)
print("✅ Processing Complete")
print(f"   SAM2 results: {Config.OUTPUT_SAM2_DIR}")
print(f"   PCD projection: {Config.OUTPUT_PCD_DIR}")
print("="*50)
