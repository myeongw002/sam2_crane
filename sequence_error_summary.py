import os
import pandas as pd
import numpy as np

def load_measurement_csv(meas_path):
    """
    measurement.csv에서 'AVERAGE' row의 4개 길이값만 추출.
    """
    df = pd.read_csv(meas_path)

    # AVERAGE row 찾기
    avg_row = df[df['frame'] == 'AVERAGE']
    if avg_row.empty:
        raise ValueError(f"AVERAGE row not found in {meas_path}")

    # 값 추출
    vals = avg_row.iloc[0][["P1-P2", "P3-P4", "P5-P6", "P7-P8"]].astype(float).values
    return vals  # shape (4,)

def compute_errors(all_seq_root="./all_sequences", gt_csv="./GT.csv"):
    """
    all_sequences 내부 디렉토리들을 탐색하여 각 measurement.csv 읽고
    GT.csv와 비교하여 오차 계산.
    """
    # GT 로드
    gt = pd.read_csv(gt_csv)
    gt.set_index("Schedule_ID", inplace=True)

    records = []

    # all_sequences 내부의 모든 디렉토리 루프
    for sched_id in sorted(os.listdir(all_seq_root)):
        sched_path = os.path.join(all_seq_root, sched_id)
        if not os.path.isdir(sched_path):
            continue

        meas_path = os.path.join(sched_path, "results", "measurements.csv")
        if not os.path.exists(meas_path):
            print(f"[WARN] measurement.csv not found: {meas_path}")
            continue

        # measurement.csv에서 평균값 추출
        try:
            meas_vals = load_measurement_csv(meas_path)
        except Exception as e:
            print(f"[ERROR] {sched_id}: {e}")
            continue

        # GT 값 가져오기
        if int(sched_id) not in gt.index:
            print(f"[WARN] Schedule ID {sched_id} not found in GT.csv – skip")
            continue

        gt_vals = gt.loc[int(sched_id), ["P1-P2", "P3-P4", "P5-P6", "P7-P8"]].values.astype(float)

        # 오차 계산
        abs_err = np.abs(meas_vals - gt_vals)     # 절대값 차이
        abs_err[0] = abs(abs_err[0]-450)  # P1-P2
        abs_err[1] = abs(abs_err[1]-450)  # P3-P4
        rel_err = abs_err / (gt_vals + 1e-6) * 100  # 백분율 오차(%)

        records.append({
            "Schedule_ID": sched_id,
            "meas_P1-P2": meas_vals[0],
            "meas_P3-P4": meas_vals[1],
            "meas_P5-P6": meas_vals[2],
            "meas_P7-P8": meas_vals[3],
            "gt_P1-P2": gt_vals[0],
            "gt_P3-P4": gt_vals[1],
            "gt_P5-P6": gt_vals[2],
            "gt_P7-P8": gt_vals[3],
            "err_P1-P2": abs_err[0],
            "err_P3-P4": abs_err[1],
            "err_P5-P6": abs_err[2],
            "err_P7-P8": abs_err[3],
            "mean_error": abs_err.mean(),
            "mean_rel_error_percent": rel_err.mean(),
        })

    result_df = pd.DataFrame(records)
    return result_df


if __name__ == "__main__":
    df_errors = compute_errors("./all_sequences", "./GT.csv")
    print(df_errors)

    df_errors.to_csv("sequence_error_summary.csv", index=False)
    print("Saved: sequence_error_summary.csv")
