import os
import re
import csv
from statistics import mean

# 설정 ----------------------------------------------------------------------
ROOT_DIR = "/workspace/sam2/crane/all_runs/"  # 실험 결과 최상위 디렉토리 (W0H0, W5H-10 등이 있는 위치)
GT_CSV_PATH = "./GT.csv"  # 질문에 올린 GT 파일 경로
OUTPUT_CSV_PATH = "./gp_dataset.csv"  # 결과 요약 CSV

# ---------------------------------------------------------------------------
# 1. GT CSV 읽기: Schedule_ID -> (P1P2, P3P4, P5P6, P7P8)
# ---------------------------------------------------------------------------
def load_ground_truth(gt_path):
    gt_dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["Schedule_ID"].strip()
            try:
                p12 = float(row["P1-P2"])
                p34 = float(row["P3-P4"])
                p56 = float(row["P5-P6"])
                p78 = float(row["P7-P8"])
            except ValueError:
                # 숫자로 변환 안되면 스킵
                continue
            gt_dict[sid] = (p12, p34, p56, p78)
    return gt_dict

# ---------------------------------------------------------------------------
# 2. measurment.csv / measurement.csv에서 평균 측정값과 DZ 추출
# ---------------------------------------------------------------------------
def parse_measurement_csv(meas_path):
    """
    measurment.csv 또는 measurement.csv 파일에서
    (avg_P1P2, avg_P3P4, avg_P5P6, avg_P7P8, DZ_mm) 를 반환.
    DZ_mm은 AVERAGE 행의 마지막 컬럼 기준.
    AVERAGE 행이 없으면, 프레임별 행을 평균내고 DZ는 숫자가 있는 행에서 하나 가져옴.
    """
    with open(meas_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r]

    header = rows[0]
    data_rows = rows[1:]

    avg_row = None
    numeric_rows = []
    dz_values = []

    for row in data_rows:
        key = row[0].strip()
        # AVERAGE 행 찾기
        if key.upper() == "AVERAGE":
            avg_row = row
            # DZ는 마지막 컬럼이라고 가정
            if len(row) >= 6 and row[5].strip() != "":
                try:
                    dz_values.append(float(row[5]))
                except ValueError:
                    pass
        else:
            # frame 번호라고 가정하고 숫자 행만 모음
            try:
                _frame_idx = int(key)
            except ValueError:
                continue
            # 길이 4개 + DZ 컬럼이 있을 수 있음
            if len(row) >= 5:
                try:
                    p12 = float(row[1])
                    p34 = float(row[2])
                    p56 = float(row[3])
                    p78 = float(row[4])
                    numeric_rows.append((p12, p34, p56, p78))
                except ValueError:
                    pass
                # DZ 컬럼이 숫자이면 수집
                if len(row) >= 6 and row[5].strip() != "":
                    try:
                        dz_values.append(float(row[5]))
                    except ValueError:
                        pass

    # 우선순위 1: AVERAGE 행이 있는 경우, 그 값을 그대로 사용
    if avg_row is not None:
        try:
            avg_p12 = float(avg_row[1])
            avg_p34 = float(avg_row[2])
            avg_p56 = float(avg_row[3])
            avg_p78 = float(avg_row[4])
        except (ValueError, IndexError):
            avg_p12 = avg_p34 = avg_p56 = avg_p78 = None
    else:
        # AVERAGE 행이 없으면 프레임별 평균 사용
        if numeric_rows:
            avg_p12 = mean([r[0] for r in numeric_rows])
            avg_p34 = mean([r[1] for r in numeric_rows])
            avg_p56 = mean([r[2] for r in numeric_rows])
            avg_p78 = mean([r[3] for r in numeric_rows])
        else:
            avg_p12 = avg_p34 = avg_p56 = avg_p78 = None

    # DZ는 수집된 값이 있으면 그 중 하나 사용, 없으면 None
    dz_mm = dz_values[0] if dz_values else None

    if None in (avg_p12, avg_p34, avg_p56, avg_p78):
        return None  # 유효하지 않은 파일
    return avg_p12, avg_p34, avg_p56, avg_p78, dz_mm

# ---------------------------------------------------------------------------
# 3. 폴더 구조 순회: W?H? / run / ScheduleID / results / measurment.csv
# ---------------------------------------------------------------------------
def extract_W_H_from_dirname(dirname):
    """
    디렉토리명 예: 'W0H0', 'W10H-5', 'W-10H10'
    정규식으로 W, H 정수값 추출
    """
    m = re.match(r"^W(-?\d+)H(-?\d+)$", dirname)
    if not m:
        return None
    W = int(m.group(1))
    H = int(m.group(2))
    return W, H

def find_measurement_files(root_dir):
    """
    root_dir 아래 전체를 돌면서
    - 상위에 'W?H?' 폴더가 있는
    - 'results/measurment.csv' 또는 'results/measurement.csv'
    파일 경로를 모두 찾는다.
    반환: 리스트 [(W, H, run_id, schedule_id, meas_csv_path), ...]
    """
    records = []
    for wdir in os.listdir(root_dir):
        wdir_path = os.path.join(root_dir, wdir)
        if not os.path.isdir(wdir_path):
            continue

        WH = extract_W_H_from_dirname(wdir)
        if WH is None:
            continue
        W, H = WH

        # W,H 디렉터리 아래: run_id (예: '001', '002', ...)
        for run_id in os.listdir(wdir_path):
            run_path = os.path.join(wdir_path, run_id)
            if not os.path.isdir(run_path):
                continue

            # run 디렉터리 아래: Schedule_ID
            for sched_id in os.listdir(run_path):
                sched_path = os.path.join(run_path, sched_id)
                if not os.path.isdir(sched_path):
                    continue

                results_path = os.path.join(sched_path, "results")
                if not os.path.isdir(results_path):
                    continue

                # measurment.csv 또는 measurement.csv 탐색
                cand1 = os.path.join(results_path, "measurments.csv")
                cand2 = os.path.join(results_path, "measurements.csv")

                if os.path.isfile(cand1):
                    meas_path = cand1
                elif os.path.isfile(cand2):
                    meas_path = cand2
                else:
                    continue

                records.append((W, H, run_id, sched_id, meas_path))

    return records

# ---------------------------------------------------------------------------
# 4. 전체 파이프라인
# ---------------------------------------------------------------------------
def main():
    gt_dict = load_ground_truth(GT_CSV_PATH)
    print(f"Loaded GT for {len(gt_dict)} schedule IDs")

    meas_files = find_measurement_files(ROOT_DIR)
    print(f"Found {len(meas_files)} measurement files")

    output_rows = []

    for W, H, run_id, sched_id, meas_path in meas_files:
        if sched_id not in gt_dict:
            # GT에 없는 Schedule_ID는 스킵
            print(f"[WARN] Schedule_ID {sched_id} not in GT. Skip.")
            continue

        parsed = parse_measurement_csv(meas_path)
        if parsed is None:
            print(f"[WARN] Invalid measurement file: {meas_path}")
            continue

        avg_p12, avg_p34, avg_p56, avg_p78, dz_mm = parsed
        gt_p12, gt_p34, gt_p56, gt_p78 = gt_dict[sched_id]

        # Error = 4개 길이 절대오차 평균
        err_p12 = abs(avg_p12 - gt_p12 - 450)  # 450mm 보정
        err_p34 = abs(avg_p34 - gt_p34 - 450)  # 450mm 보정
        err_p56 = abs(avg_p56 - gt_p56)
        err_p78 = abs(avg_p78 - gt_p78)
        err_mean = (err_p12 + err_p34 + err_p56 + err_p78) / 4.0

        output_rows.append({
            "Schedule_ID": str(sched_id),
            "Run_ID": run_id,
            "W": W,
            "H": H,
            "DZ_mm": dz_mm,
            "Err_mean_mm": err_mean,
            "Err_P1P2_mm": err_p12,
            "Err_P3P4_mm": err_p34,
            "Err_P5P6_mm": err_p56,
            "Err_P7P8_mm": err_p78,
        })

    # 결과 저장
    fieldnames = [
        "Schedule_ID",
        "Run_ID",
        "W",
        "H",
        "DZ_mm",
        "Err_mean_mm",
        "Err_P1P2_mm",
        "Err_P3P4_mm",
        "Err_P5P6_mm",
        "Err_P7P8_mm",
    ]

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Saved {len(output_rows)} rows to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
