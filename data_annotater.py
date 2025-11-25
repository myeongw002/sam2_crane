import os
import glob
import cv2
import csv
import re
import numpy as np
import open3d as o3d

# ============================
# 설정
# ============================
ROOT_DIR     = "/workspace/sequences_sample/20251123"  # 스케줄 ID 폴더들이 있는 루트
DATASET_CSV  = "./annotation/dataset.csv"

INTRINSIC_PATH = "/workspace/sam2/intrinsic.csv"
EXTRINSIC_PATH = "/workspace/sam2/transform3_tuned_tuned.txt"

# ============================
# 로그 파싱 정규식
# ============================
DZ_PATTERN      = re.compile(r"DZ Distance\s*:\s*(\d+)\s*mm")
WIDTH_PATTERN   = re.compile(r"Plate Max Width\s*:\s*(\d+)\s*mm")
TOPLEN_PATTERN  = re.compile(r"Plate Top Length\s*:\s*(\d+)\s*mm")


def parse_results_txt(txt_path):
    """results 안 txt에서 DZ / MaxWidth / TopLength 읽기"""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    dz = DZ_PATTERN.search(content)
    w  = WIDTH_PATTERN.search(content)
    L  = TOPLEN_PATTERN.search(content)

    if not (dz and w and L):
        raise ValueError(f"파싱 실패: {txt_path}")

    return int(dz.group(1)), int(w.group(1)), int(L.group(1))


def ensure_csv_dir():
    """DATASET_CSV 상위 디렉토리가 없으면 생성"""
    dir_path = os.path.dirname(DATASET_CSV)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


# ============================
# 카메라 파라미터 로드
# ============================
def load_camera_params(intrinsic_path, extrinsic_path):
    """카메라 내/외부 파라미터 로드 (질문에서 주신 형식 사용)"""
    # Intrinsic
    intrinsic = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
    K = np.array([
        [intrinsic[0], intrinsic[1], intrinsic[2]],
        [0.0,          intrinsic[3], intrinsic[4]],
        [0.0,          0.0,          1.0]
    ], dtype=np.float32)

    fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]
    D = np.array(
        [intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]],
        dtype=np.float32
    )

    # Extrinsic: LiDAR → Camera
    T_l2c = np.loadtxt(extrinsic_path, delimiter=',').astype(np.float32)

    print("[INFO] Camera parameters loaded")
    print(f"   Intrinsic: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print("   Extrinsic (LiDAR to Camera):")
    print(T_l2c)
    return K, D, T_l2c


# ============================
# PCD → 이미지 투영 (DZ 기반 depth 필터)
# ============================
def overlay_pcd_with_depth_filter(img, points_lidar, dz_mm, K, D, T_l2c):
    """
    이미지 위에, DZ 기반 depth 범위 (10 - dz[m] + 0.3 ± 0.1)에 해당하는 점들만 투영하여 오버레이.
    - img: (H, W, 3) BGR
    - points_lidar: (N, 3) LiDAR 좌표계 포인트
    - dz_mm: DZ Distance [mm]
    """

    if points_lidar.size == 0:
        return img

    H, W = img.shape[:2]

    R = T_l2c[:3, :3]
    t = T_l2c[:3, 3]

    # LiDAR → Camera
    pts_cam = (R @ points_lidar.T).T + t  # (N,3)
    Z = pts_cam[:, 2]

    # DZ 기반 depth center 계산
    dz_m = dz_mm * 0.001
    depth_center = 10.0 - dz_m + 0.3
    z_min = depth_center - 0.1
    z_max = depth_center + 0.1

    depth_mask = (Z > 1e-6) & (Z >= z_min) & (Z <= z_max)
    pts_cam_sel = pts_cam[depth_mask]

    if pts_cam_sel.shape[0] == 0:
        print(f"[INFO] depth range에 해당하는 포인트 없음 (z_min={z_min:.3f}, z_max={z_max:.3f})")
        return img

    # 카메라 좌표계 3D 포인트를 직접 projectPoints에 넣기 위해 rvec, tvec=0 사용
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    # (N,1,3) 형태로 reshape
    pts_cam_input = pts_cam_sel.reshape(-1, 1, 3).astype(np.float32)

    img_pts, _ = cv2.projectPoints(pts_cam_input, rvec, tvec, K, D)
    img_pts = img_pts.reshape(-1, 2)
    u = img_pts[:, 0]
    v = img_pts[:, 1]

    # 이미지 범위 필터
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_img].astype(np.int32)
    v = v[in_img].astype(np.int32)
    pts_cam_sel = pts_cam_sel[in_img]

    if u.size == 0:
        print("[INFO] 투영 후 이미지 범위 안에 포인트 없음")
        return img

    # 깊이(Z) 기준으로 Turbo 컬러맵
    z_vals = pts_cam_sel[:, 2]
    z_min_v, z_max_v = z_vals.min(), z_vals.max()
    if z_max_v - z_min_v < 1e-6:
        norm = np.zeros_like(z_vals, dtype=np.uint8)
    else:
        norm = ((z_vals - z_min_v) / (z_max_v - z_min_v) * 255).astype(np.uint8)

    norm_img = norm.reshape(-1, 1, 1)
    colors = cv2.applyColorMap(norm_img, cv2.COLORMAP_TURBO).reshape(-1, 3)

    overlay = img.copy()
    for px, py, color in zip(u, v, colors):
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        cv2.circle(overlay, (px, py), 2, (b, g, r), -1)

    return overlay


# ============================
# 마우스 콜백 (프롬프트 2~4개)
# ============================
points = []  # [(x,y), ...]

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 4:
            print("이미 4개까지 선택했습니다. (r로 초기화 가능)")
            return
        points.append((x, y))
        print(f"Clicked #{len(points)}: ({x}, {y})")


# ============================
# 메인 루프
# ============================
def main():
    global points

    # CSV 상위 디렉토리 보장
    ensure_csv_dir()

    # dataset 파일 존재 여부 체크
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    file_exists = os.path.exists(DATASET_CSV)
    csv_file = open(DATASET_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow([
            "schedule_id", "img_path", "pcd_path", "txt_path",
            "dz_mm", "max_width_mm", "top_length_mm",
            "point_idx", "u", "v"
        ])

    # 카메라 파라미터 1회 로드
    K, D, T_l2c = load_camera_params(INTRINSIC_PATH, EXTRINSIC_PATH)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_callback)

    # 스케줄 ID 폴더 찾기
    schedule_dirs = sorted([
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
           and d.isdigit()
    ])

    for schedule_id in schedule_dirs:
        sched_path   = os.path.join(ROOT_DIR, schedule_id)
        image_dir    = os.path.join(sched_path, "image")
        pcd_dir      = os.path.join(sched_path, "pcd")
        results_dir  = os.path.join(sched_path, "results")

        img_path = os.path.join(image_dir, "0000.jpg")
        pcd_path = os.path.join(pcd_dir,   "0000.pcd")

        if not os.path.exists(img_path):
            print(f"[SKIP] 이미지 없음: {img_path}")
            continue
        if not os.path.exists(pcd_path):
            print(f"[SKIP] PCD 없음: {pcd_path}")
            continue

        txt_list = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
        if not txt_list:
            print(f"[SKIP] txt 없음: {results_dir}")
            continue
        txt_path = txt_list[0]

        dz_mm, W_mm, L_mm = parse_results_txt(txt_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] 이미지 로드 실패: {img_path}")
            continue

        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            print(f"[SKIP] PCD 로드 실패 또는 empty: {pcd_path}")
            continue
        points_lidar = np.asarray(pcd.points, dtype=np.float32)

        # DZ 기반 depth 필터를 적용한 PCD 오버레이 이미지 생성
        base_img = overlay_pcd_with_depth_filter(img, points_lidar, dz_mm, K, D, T_l2c)

        print("=" * 70)
        print(f"Schedule: {schedule_id}")
        print(f"  Image : {img_path}")
        print(f"  PCD   : {pcd_path}")
        print(f"  Log   : {txt_path}")
        print(f"  DZ={dz_mm} mm | W={W_mm} mm | L={L_mm} mm")
        print("   왼쪽 클릭: 프롬프트 포인트 추가 (최소 2개, 최대 4개)")
        print("   r: 현재 이미지에서 포인트 전부 초기화")
        print("   s: 저장 (2~4개일 때만 저장)")
        print("   n: 이 스케줄 스킵")
        print("   q: 전체 종료")

        points = []

        while True:
            draw = base_img.copy()
            # 클릭된 모든 포인트 그리기
            for i, (x, y) in enumerate(points, start=1):
                cv2.circle(draw, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(draw, str(i), (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("image", draw)

            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                csv_file.close()
                cv2.destroyAllWindows()
                print("종료")
                return

            if key == ord('r'):
                points = []
                print("현재 이미지 포인트 초기화")

            if key == ord('n'):
                print("스케줄 스킵")
                break

            if key == ord('s'):
                if not (2 <= len(points) <= 4):
                    print("저장하려면 최소 2개, 최대 4개를 선택해야 합니다.")
                    continue
                # 각 포인트를 별도 row로 저장, point_idx = 1..len(points)
                for idx, (u, v) in enumerate(points, start=1):
                    writer.writerow([
                        schedule_id, img_path, pcd_path, txt_path,
                        dz_mm, W_mm, L_mm,
                        idx, u, v
                    ])
                csv_file.flush()
                print(f"{len(points)}개 포인트 저장 완료")
                break

    csv_file.close()
    cv2.destroyAllWindows()
    print("모든 스케줄 처리 완료")


if __name__ == "__main__":
    main()
