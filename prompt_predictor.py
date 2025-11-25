import os
import glob
import re
import joblib
import cv2
import numpy as np

MODEL_PATH = "./annotation/model_prompt.pkl"

# 2 ~ 4 사이에서 원하는 개수 설정 가능
N_POINTS = 4

# ---------------------------------------------
# 로그 파싱 정규식
# ---------------------------------------------
DZ_PATTERN = re.compile(r"DZ Distance\s*:\s*(\d+)\s*mm")
WIDTH_PATTERN = re.compile(r"Plate Max Width\s*:\s*(\d+)\s*mm")
TOPLEN_PATTERN = re.compile(r"Plate Top Length\s*:\s*(\d+)\s*mm")

def parse_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()
    dz = int(DZ_PATTERN.search(t).group(1))
    w  = int(WIDTH_PATTERN.search(t).group(1))
    L  = int(TOPLEN_PATTERN.search(t).group(1))
    return dz, w, L


# ---------------------------------------------
# 프롬프트 예측
# ---------------------------------------------
def predict_for_schedule(schedule_id, n_points=N_POINTS):
    if not (2 <= n_points <= 4):
        raise ValueError("n_points는 2~4 사이여야 합니다.")

    model = joblib.load(MODEL_PATH)

    base_dir = f"/workspace/sequences_sample/{schedule_id}"
    results_dir = os.path.join(base_dir, "results")
    image_dir   = os.path.join(base_dir, "image")

    txts = sorted(glob.glob(os.path.join(results_dir, "*.txt")))
    if not txts:
        raise ValueError(f"txt 없음: {results_dir}")
    txt_path = txts[0]

    img_path = os.path.join(image_dir, "0000.jpg")
    if not os.path.exists(img_path):
        raise ValueError(f"이미지 없음: {img_path}")

    dz, w, L = parse_txt(txt_path)

    prompts = []
    for point_idx in range(1, n_points+1):
        X_in = np.array([[dz, w, L, point_idx]], dtype=float)
        u_pred, v_pred = model.predict(X_in)[0]
        prompts.append((int(round(u_pred)), int(round(v_pred))))

    return prompts, img_path


# ---------------------------------------------
# 이미지에 예측된 프롬프트 그리기 + 저장 개선
# ---------------------------------------------
def draw_and_save_image(img_path, prompts, schedule_id):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지 로드 실패: {img_path}")

    for i, (u, v) in enumerate(prompts, start=1):
        cv2.circle(img, (u, v), 8, (0, 0, 255), -1)
        cv2.putText(img, str(i), (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)

    # ============================
    # 저장 경로 개선
    # ============================
    save_dir = f"./prompt_prediction/{schedule_id}"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"predicted_prompt_overlay_{len(prompts)}points.jpg"
    out_path = os.path.join(save_dir, filename)

    cv2.imwrite(out_path, img)
    print(f"[SAVE] 결과 이미지 저장: {out_path}")

    return out_path


# ---------------------------------------------
# 메인 실행
# ---------------------------------------------
if __name__ == "__main__":
    sid = "202511201026498860"

    prompts, img_path = predict_for_schedule(sid, n_points=N_POINTS)
    print(f"Schedule {sid}: predicted prompts = {prompts}")

    draw_and_save_image(img_path, prompts, sid)
