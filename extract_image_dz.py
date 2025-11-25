import os
import re
import shutil
import cv2
import numpy as np  # 추가

# -------------------------------
# 설정
# -------------------------------
ROOT_DIR = "/workspace/sequences_sample/20251123"  # 스케줄 ID 폴더들이 있는 루트 디렉토리
OUTPUT_DIR = "./dz_images"                         # 출력 폴더
IMAGE_EXTS = (".jpg", ".jpeg", ".png")             # 이미지 확장자

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DZ Distance 추출용 정규식
DZ_PATTERN = re.compile(r"DZ Distance\s*:\s*(\d+)\s*mm")

def extract_dz(txt_path):
    """텍스트 파일에서 DZ Distance 정수 추출"""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = DZ_PATTERN.search(content)
    return int(m.group(1)) if m else None

def get_first_image(img_dir):
    """image 폴더에서 첫 번째 이미지 파일 경로 반환"""
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)])
    if not imgs:
        return None
    return os.path.join(img_dir, imgs[0])

def get_first_txt(results_dir):
    """results 폴더에서 첫 번째 txt 파일 경로 반환"""
    txts = sorted([f for f in os.listdir(results_dir) if f.lower().endswith(".txt")])
    if not txts:
        return None
    return os.path.join(results_dir, txts[0])

def unique_outpath(base_path):
    """이름 중복 시 _1, _2 ... 자동 생성"""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    n = 1
    while True:
        np_ = f"{root}_{n}{ext}"
        if not os.path.exists(np_):
            return np_
        n += 1

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

# -------------------------------
# 메인
# -------------------------------
for sched_id in sorted(os.listdir(ROOT_DIR)):
    sched_path = os.path.join(ROOT_DIR, sched_id)
    if not os.path.isdir(sched_path):
        continue

    image_dir = os.path.join(sched_path, "image")
    results_dir = os.path.join(sched_path, "results")

    if not (os.path.isdir(image_dir) and os.path.isdir(results_dir)):
        continue

    print(f"\n[INFO] Processing schedule: {sched_id}")

    # 텍스트 하나만
    txt_path = get_first_txt(results_dir)
    if not txt_path:
        print("  - No txt file found.")
        continue

    dz = extract_dz(txt_path)
    if dz is None:
        print("  - DZ Distance not found.")
        continue

    # depth 계산 (기존 로직 유지)
    depth = 10000 - dz + 300

    # 이미지 하나만
    img_path = get_first_image(image_dir)
    if not img_path:
        print("  - No image file found.")
        continue

    # 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print("  - Failed to read image.")
        continue

    h, w = img.shape[:2]

    # dz 값으로부터 프롬프트 포인트 자동 계산
    prompt_points = get_prompt_points_from_dz(dz)  # shape (3,2)

    # 프롬프트 포인트에 점 찍기
    for (x_f, y_f) in prompt_points:
        x = int(round(x_f))
        y = int(round(y_f))
        if 0 <= x < w and 0 <= y < h:
            # 빨간색 십자 마커
            cv2.drawMarker(
                img,
                (x, y),
                color=(0, 0, 255),           # BGR: red
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        else:
            print(f"  - Point ({x}, {y}) is outside image bounds ({w}x{h}), skipped.")

    # 출력 파일 이름 = depth.jpg (중복 시 자동 증가)
    out_path = os.path.join(OUTPUT_DIR, f"{depth}.jpg")
    out_path = unique_outpath(out_path)

    # 점이 찍힌 이미지 저장
    cv2.imwrite(out_path, img)

    print(f"  - Saved with markers: {img_path} → {os.path.basename(out_path)} (dz={dz})")

print("\n[DONE] All schedules processed.")
