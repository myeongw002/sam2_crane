import numpy as np
import cv2
import open3d as o3d

# -------------------------------------------------------------------
# 0. 설정 부분 (경로만 본인 환경에 맞게 수정)
# -------------------------------------------------------------------
INTRINSIC_PATH = "/workspace/sam2/intrinsic.csv"
EXTRINSIC_PATH = "/workspace/sam2/transform3_tuned_tuned.txt"
IMAGE_PATH     = "/workspace/sequences_sample/202511201026498853/image/0004.jpg"
PCD_PATH       = "/workspace/sequences_sample/202511201026498853/pcd/0004.pcd"

ACWL_DZ = 1856  # mm, 카메라-라이다 수직 거리
MAX_DEPTH = 10 - (ACWL_DZ * 0.001) + 0.5  # m
# MAX_DEPTH = 8.5

# -------------------------------------------------------------------
# 1. 카메라 파라미터 로드 (질문에 주신 코드 그대로 사용)
# -------------------------------------------------------------------
def load_camera_params(intrinsic_path, extrinsic_path):
    """카메라 내/외부 파라미터 로드"""
    # Intrinsic
    intrinsic = np.loadtxt(intrinsic_path, delimiter=',', usecols=range(9))
    K = np.array([
        [intrinsic[0], intrinsic[1], intrinsic[2]],
        [0.0,          intrinsic[3], intrinsic[4]],
        [0.0,          0.0,          1.0]
    ], dtype=np.float32)
    
    fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]
    D = np.array([intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]], dtype=np.float32)
    # Extrinsic: LiDAR → Camera
    T_l2c = np.loadtxt(extrinsic_path, delimiter=',').astype(np.float32)
    
    print("[INFO] Camera parameters loaded")
    print(f"   Intrinsic: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print("   Extrinsic (LiDAR to Camera):")
    print(T_l2c)
    return K, D, T_l2c

# -------------------------------------------------------------------
# 2. Extrinsic: RPY <-> 4x4 변환
# -------------------------------------------------------------------
def euler_from_matrix(R):
    """
    회전행렬 R (3x3) → roll, pitch, yaw [rad]
    ZYX 순서 (Rz * Ry * Rx) 기준
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:
        # 거의 시니귤러한 경우
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = 0.0
    return roll, pitch, yaw

def matrix_from_euler(roll, pitch, yaw):
    """
    roll, pitch, yaw [rad] → 회전행렬 R (3x3)
    ZYX 순서 (Rz * Ry * Rx)
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return R

def decompose_extrinsic(T):
    """
    4x4 extrinsic (LiDAR→Camera) → tx,ty,tz [m], roll,pitch,yaw [deg]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    roll, pitch, yaw = euler_from_matrix(R)
    return t[0], t[1], t[2], np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def compose_extrinsic(tx, ty, tz, roll_deg, pitch_deg, yaw_deg):
    """
    tx,ty,tz [m], roll,pitch,yaw [deg] → 4x4 extrinsic (LiDAR→Camera)
    """
    roll  = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw   = np.radians(yaw_deg)
    R = matrix_from_euler(roll, pitch, yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = np.array([tx, ty, tz], dtype=np.float32)
    return T

def quat_normalize(q):
    """q = [w, x, y, z]"""
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n


def quat_mul(q2, q1):
    """
    q = q2 * q1  (둘 다 [w, x, y, z])
    전역축 기준 증분 회전하려면 q_new = delta * q_current 로 사용.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1
    return quat_normalize([w, x, y, z])


def quat_from_axis_angle(axis, angle_rad):
    """
    axis: (3,) 단위벡터, angle_rad: 라디안
    """
    axis = np.asarray(axis, dtype=np.float32)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = axis / n
    half = angle_rad * 0.5
    s = np.sin(half)
    w = np.cos(half)
    x, y, z = axis * s
    return quat_normalize([w, x, y, z])


def quat_to_rotmat(q):
    """q = [w, x, y, z] → 3x3 회전행렬"""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),         2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float32)
    return R


def rotmat_to_quat(R):
    """
    3x3 회전행렬 → q = [w, x, y, z]
    """
    R = np.asarray(R, dtype=np.float32)
    tr = R[0,0] + R[1,1] + R[2,2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S

    return quat_normalize([w, x, y, z])


# -------------------------------------------------------------------
# 3. LiDAR 포인트 → 이미지 투영 (질문 코드 재사용)
# -------------------------------------------------------------------
def project_pcd_to_image(points_lidar, T_l2c, K, dist_coeffs, W, H, max_depth=10.0):
    """
    LiDAR 포인트를 이미지로 투영 (Why곡 보정 포함)
    K: (3,3) Intrinsic Matrix
    dist_coeffs: (4,) or (5,) Distortion coefficients
    """
    # 1. Z값 필터링 (카메라 좌표계 변환 후 수행)
    # 먼저 회전/이동만 적용하여 카메라 좌표계로 변환
    R = T_l2c[:3, :3]
    t = T_l2c[:3, 3]
    
    # (N, 3)
    pts_cam = (R @ points_lidar.T).T + t
    
    # 깊이 필터링
    Z = pts_cam[:, 2]
    valid_mask = (Z > 1e-6) & (Z < max_depth)
    
    pts_cam_valid = pts_cam[valid_mask]
    
    if len(pts_cam_valid) == 0:
        return np.array([]), np.array([]), np.zeros(len(points_lidar), dtype=bool), pts_cam

    # 2. cv2.projectPoints를 사용하여 투영 (Why곡 보정 적용됨)
    # rvec은 회전 행렬을 로드리게스 벡터로 변환해야 함 (여기서는 이미 변환된 pts_cam을 쓰므로 rvec=0, tvec=0으로 두고 3D 점을 직접 넣는 트릭을 쓰거나,
    # 정석대로 원래 points_lidar[valid_mask]를 넣고 rvec, tvec를 넣어야 함. 정석 방법 추천)
    
    rvec, _ = cv2.Rodrigues(R)
    tvec = t
    
    # 입력: (N, 1, 3) 형태여야 함
    img_pts, _ = cv2.projectPoints(points_lidar[valid_mask], rvec, tvec, K, dist_coeffs)
    img_pts = img_pts.squeeze() # (N, 2)
    
    u = img_pts[:, 0]
    v = img_pts[:, 1]
    
    # 3. 이미지 범위 필터링
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # valid_mask 중에서 in_img인 것만 최종 True
    # 인덱싱이 복잡해지므로, 결과를 재구성하는 로직이 필요합니다.
    # 편의상 유효한 u, v만 리턴하거나, 전체 사이즈 마스크를 리턴하도록 구조를 맞춰야 합니다.
    
    # (간단한 수정을 위해 기존 구조 유지하되 projectPoints만 적용하는 방식)
    # 다만 cv2.projectPoints는 왜곡 때문에 u,v 계산식이 복잡하므로 위 함수로 대체하는 것이 좋습니다.
    
    final_valid_indices = np.where(valid_mask)[0][in_img]
    final_valid_mask = np.zeros(len(points_lidar), dtype=bool)
    final_valid_mask[final_valid_indices] = True
    
    # 전체 배열 크기에 맞춰 u, v 확장 (유효하지 않은 곳은 0)
    u_full = np.zeros(len(points_lidar), dtype=np.float32)
    v_full = np.zeros(len(points_lidar), dtype=np.float32)
    
    u_full[final_valid_indices] = u[in_img]
    v_full[final_valid_indices] = v[in_img]
    
    return u_full, v_full, final_valid_mask, pts_cam
# -------------------------------------------------------------------
# 4. 시각화: 이미지 위에 포인트 오버레이
# -------------------------------------------------------------------
def draw_single_axis_colormap(image, u, v, valid_mask, pts_cam, mode="z"):
    """
    mode: "x" / "y" / "z" 중 하나
    선택한 축의 값을 정규화하여 Turbo 컬러맵으로 색을 입힘
    """
    img_overlay = image.copy()

    u_valid = u[valid_mask].astype(np.int32)
    v_valid = v[valid_mask].astype(np.int32)
    pts_valid = pts_cam[valid_mask]

    # ─────────────────────────────────────────────
    # 1) 선택된 축 값 추출
    # ─────────────────────────────────────────────
    if mode == "x":
        vals = pts_valid[:, 0]
    elif mode == "y":
        vals = pts_valid[:, 1]
    elif mode == "z":
        vals = pts_valid[:, 2]
    else:
        raise ValueError("mode must be 'x', 'y', or 'z'.")

    # ─────────────────────────────────────────────
    # 2) 값 정규화 (0~255)
    # ─────────────────────────────────────────────
    vmin, vmax = np.min(vals), np.max(vals)
    if vmax - vmin < 1e-6:
        norm = np.zeros_like(vals, dtype=np.uint8)
    else:
        norm = ((vals - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # (N,1,1) shape으로 변경 후 Turbo 컬러맵 적용
    norm_img = norm.reshape(-1, 1, 1)
    colors = cv2.applyColorMap(norm_img, cv2.COLORMAP_TURBO).reshape(-1, 3)

    # ─────────────────────────────────────────────
    # 3) 점 찍기
    # ─────────────────────────────────────────────
    for px, py, color in zip(u_valid, v_valid, colors):
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        cv2.circle(img_overlay, (px, py), 3, (b, g, r), -1)

    return img_overlay


# -------------------------------------------------------------------
# 5. Extrinsic 수동 조절 인터랙티브 루프
# -------------------------------------------------------------------

def interactive_adjust_extrinsic():
    # (1) 카메라 파라미터 & 초기 extrinsic 로드
    K, D, T_init = load_camera_params(INTRINSIC_PATH, EXTRINSIC_PATH)

    # (2) 이미지 로드
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {IMAGE_PATH}")
    H, W = img.shape[:2]

    # (3) LiDAR 포인트 로드
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    if pcd.is_empty():
        raise FileNotFoundError(f"PCD 파일을 열 수 없습니다: {PCD_PATH}")
    points_lidar = np.asarray(pcd.points, dtype=np.float32)

    # (4) extrinsic 초기값 (t, q) 설정
    R0 = T_init[:3, :3]
    t0 = T_init[:3, 3]
    q = rotmat_to_quat(R0)          # 현재 쿼터니언
    tx, ty, tz = float(t0[0]), float(t0[1]), float(t0[2])

    # 리셋용 백업
    q_init = q.copy()
    init_tx, init_ty, init_tz = tx, ty, tz

    # (5) 스텝 설정
    trans_step = 0.01   # m
    rot_step   = 0.1    # deg

    # 전역 축 (LiDAR 기준)
    axis_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # roll 축
    axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # pitch 축
    axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # yaw 축

    def print_help():
        print("\n[KEY GUIDE]")
        print("  이동 (Translation):")
        print(f"    q/a : tx +/− (step={trans_step:.4f} m)")
        print(f"    w/s : ty +/− (step={trans_step:.4f} m)")
        print(f"    e/d : tz +/− (step={trans_step:.4f} m)\n")

        print("  회전 (Rotation, global axes):")
        print(f"    r/f : roll(X축)  +/− (step={rot_step:.3f} deg)")
        print(f"    t/g : pitch(Y축) +/− (step={rot_step:.3f} deg)")
        print(f"    y/h : yaw(Z축)   +/− (step={rot_step:.3f} deg)\n")

        print("  기능 (숫자키):")
        print("    1 : 도움말 다시 출력")
        print("    2 : extrinsic 파라미터 리셋")
        print("    3 : translation step 변경")
        print("    4 : rotation step 변경")
        print("    5 : extrinsic 저장 후 종료")
        print("    6 : 종료")
        print("    SPACE : 재렌더링")

    print_help()
    print(f"[INFO] 초기 step - trans_step={trans_step:.4f} m, rot_step={rot_step:.3f} deg")

    cv2.namedWindow("Lidar Projection", cv2.WINDOW_NORMAL)

    while True:
        # 현재 쿼터니언 → 회전행렬
        R = quat_to_rotmat(q)
        T_l2c = np.eye(4, dtype=np.float32)
        T_l2c[:3, :3] = R
        T_l2c[:3, 3]  = np.array([tx, ty, tz], dtype=np.float32)

        # 포인트 투영
        u, v, valid, pts_cam = project_pcd_to_image(
            points_lidar, T_l2c, K, D,  W, H, max_depth=MAX_DEPTH
        )

        # 오버레이 이미지
        overlay = draw_single_axis_colormap(img, u, v, valid, pts_cam, mode="z")

        # 표시용 rpy (쎄한 구간에서도 그냥 표시용이라 상관 없음)
        roll_rad, pitch_rad, yaw_rad = euler_from_matrix(R)
        roll_deg = np.degrees(roll_rad)
        pitch_deg = np.degrees(pitch_rad)
        yaw_deg = np.degrees(yaw_rad)

        info = f"t=({tx:.3f},{ty:.3f},{tz:.3f})  rpy=({roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 2, 2
        (text_w, text_h), baseline = cv2.getTextSize(info, font, font_scale, thickness)
        x, y = 20, 20 + text_h
        cv2.putText(overlay, info, (x, y), font, font_scale, (0,255,0), thickness, cv2.LINE_AA)

        cv2.imshow("Lidar Projection", overlay)
        key = cv2.waitKey(0) & 0xFF

        # ───────── 숫자 기능키 ─────────
        if key == ord('1'):          # 도움말
            print_help()
            continue

        elif key == ord('2'):        # 리셋
            q = q_init.copy()
            tx, ty, tz = init_tx, init_ty, init_tz
            print("[INFO] Reset to initial extrinsic.")
            continue

        elif key == ord('3'):        # translation step 변경
            try:
                new_step = float(input(f"[INPUT] New translation step (현재 {trans_step:.6f} m): "))
                if new_step > 0:
                    trans_step = new_step
                print(f"[INFO] trans_step={trans_step:.6f} m, rot_step={rot_step:.6f} deg")
            except Exception as e:
                print(f"[WARN] 잘못된 입력: {e}")
            continue

        elif key == ord('4'):        # rotation step 변경
            try:
                new_step = float(input(f"[INPUT] New rotation step (현재 {rot_step:.6f} deg): "))
                if new_step > 0:
                    rot_step = new_step
                print(f"[INFO] trans_step={trans_step:.6f} m, rot_step={rot_step:.6f} deg")
            except Exception as e:
                print(f"[WARN] 잘못된 입력: {e}")
            continue

        elif key == ord('5'):        # 저장 후 종료
            R_save = quat_to_rotmat(q)
            T_save = np.eye(4, dtype=np.float32)
            T_save[:3, :3] = R_save
            T_save[:3, 3]  = np.array([tx, ty, tz], dtype=np.float32)
            # 확장자 제거 후 _tuned 추가
            base_path = EXTRINSIC_PATH.rsplit('.txt', 1)[0]
            save_path = f"{base_path}_tuned.txt"
            np.savetxt(save_path, T_save, delimiter=',', fmt="%.8f")
            print(f"[INFO] Saved tuned extrinsic to: {save_path}")
            break

        elif key == ord('6'):        # 종료
            print("[INFO] Quit without saving.")
            break

        # ───────── Translation (q/a, w/s, e/d) ─────────
        elif key == ord('q'):        # tx +
            tx += trans_step
        elif key == ord('a'):        # tx -
            tx -= trans_step

        elif key == ord('w'):        # ty +
            ty += trans_step
        elif key == ord('s'):        # ty -
            ty -= trans_step

        elif key == ord('e'):        # tz +
            tz += trans_step
        elif key == ord('d'):        # tz -
            tz -= trans_step

        # ───────── Rotation (r/f, t/g, y/h) 쿼터니언 증분 ─────────
        elif key == ord('r'):        # roll + (X축)
            angle = np.radians(rot_step)
            dq = quat_from_axis_angle(axis_x, angle)
            q = quat_mul(dq, q)
        elif key == ord('f'):        # roll - (X축)
            angle = -np.radians(rot_step)
            dq = quat_from_axis_angle(axis_x, angle)
            q = quat_mul(dq, q)

        elif key == ord('t'):        # pitch + (Y축)
            angle = np.radians(rot_step)
            dq = quat_from_axis_angle(axis_y, angle)
            q = quat_mul(dq, q)
        elif key == ord('g'):        # pitch - (Y축)
            angle = -np.radians(rot_step)
            dq = quat_from_axis_angle(axis_y, angle)
            q = quat_mul(dq, q)

        elif key == ord('y'):        # yaw + (Z축)
            angle = np.radians(rot_step)
            dq = quat_from_axis_angle(axis_z, angle)
            q = quat_mul(dq, q)
        elif key == ord('h'):        # yaw - (Z축)
            angle = -np.radians(rot_step)
            dq = quat_from_axis_angle(axis_z, angle)
            q = quat_mul(dq, q)

        # SPACE: 재렌더
        elif key == ord(' '):
            pass

        else:
            print(f"[INFO] Unknown key: {key}")
            continue

        print(f"  -> t=({tx:.4f},{ty:.4f},{tz:.4f}), rpy=({roll_deg:.3f},{pitch_deg:.3f},{yaw_deg:.3f})")
        print(f"     [step] trans_step={trans_step:.6f} m, rot_step={rot_step:.6f} deg")

    cv2.destroyAllWindows()

# -------------------------------------------------------------------
# 6. 실행
# -------------------------------------------------------------------
if __name__ == "__main__":
    interactive_adjust_extrinsic()
