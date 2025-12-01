# gp_dataset_process_and_plot.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# 1) GP 데이터셋 정리 (run 평균 등)
# ===============================

def load_and_clean_gp_dataset(
    input_csv="gp_dataset.csv",
    output_csv="gp_dataset_clean.csv",
    schedule_col="Schedule_ID",
    dz_col="DZ_mm",
    w_col="W",
    h_col="H",
):
    """
    gp_dataset.csv를 불러와서 다음을 수행:
      1) 에러 컬럼 자동 탐지 (이름이 'Err_'로 시작하는 컬럼)
      2) (Schedule_ID, DZ, W, H) 별로 평균을 내서 run 간 노이즈를 줄임
      3) 필요시 Err_mean_mm 컬럼 생성
      4) gp_dataset_clean.csv 로 저장

    반환값:
      df_clean: 정리된 DataFrame
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} 파일을 찾을 수 없습니다.")

    df = pd.read_csv(input_csv)

    # ---- 에러 컬럼 자동 탐지 ----
    # 예: Err_P1P2, Err_P3P4, ..., Err_mean_mm 등
    error_cols = [c for c in df.columns if c.startswith("Err_")]

    if len(error_cols) == 0:
        raise ValueError("컬럼 이름이 'Err_'로 시작하는 에러 컬럼을 찾을 수 없습니다.")

    # ---- Err_mean_mm 없으면 생성 ----
    if "Err_mean_mm" not in df.columns:
        # P1P2, P3P4, P5P6, P7P8 각각의 에러가 들어있다고 가정하고 평균
        # (실제 컬럼 이름과 다르면 여기만 수정하면 됨)
        # Err_mean_mm이 이미 있으면 이 단계는 건너뜀.
        # 아래는 '측정값 - GT' 절댓값을 이미 Err_P1P2 이런 컬럼으로 넣었다고 보고,
        # 그 4개 평균을 Err_mean_mm으로 만든다는 의미.
        side_error_cols = [c for c in error_cols if c != "Err_mean_mm"]
        if len(side_error_cols) > 0:
            df["Err_mean_mm"] = df[side_error_cols].mean(axis=1)
            error_cols = [c for c in df.columns if c.startswith("Err_")]  # 갱신

    # ---- 그룹핑 기준 ----
    group_cols = [schedule_col, dz_col, w_col, h_col]

    # 그룹핑 가능한지 체크
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' 컬럼이 입력 CSV에 없습니다. 컬럼 이름을 확인해 주세요.")

    # ---- 그룹별 평균 (run 여러 번 한 경우 평균) ----
    agg_dict = {col: "mean" for col in error_cols}
    df_clean = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # 저장
    df_clean.to_csv(output_csv, index=False)
    print(f"[INFO] 정리된 GP 데이터셋을 '{output_csv}'로 저장했습니다.")
    print(f"[INFO] 총 row 수: {len(df_clean)}")
    return df_clean


# ===============================
# 2) 전체 스케줄 ID에 대한 시각화
# ===============================

def plot_dz_vs_error_by_WH(
    df_clean,
    dz_col="DZ_mm",
    w_col="W",
    h_col="H",
    err_col="Err_mean_mm",
    save_path=None,
):
    """
    전체 스케줄을 한 번에 보면서, DZ - Error 관계를 W, H에 따라 보는 산점도.

    - x축: DZ_mm
    - y축: Err_mean_mm
    - 색상: W
    - 마커 크기: H (상대값)
    """

    if err_col not in df_clean.columns:
        raise ValueError(f"'{err_col}' 컬럼이 없습니다.")

    # H값을 이용해 마커 크기 스케일링
    H_vals = df_clean[h_col].values
    H_min = H_vals.min()
    # 너무 작거나 0이 되지 않게 offset 추가
    marker_sizes = (H_vals - H_min + 1) * 10.0

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df_clean[dz_col],
        df_clean[err_col],
        c=df_clean[w_col],
        s=marker_sizes,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    plt.colorbar(sc, label=f"{w_col} 값")
    plt.xlabel(dz_col)
    plt.ylabel(err_col)
    plt.title("DZ vs Error (색: W, 크기: H)")

    plt.grid(True, linestyle="--", alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] DZ vs Error 플롯을 '{save_path}'에 저장했습니다.")
    else:
        plt.show()


def plot_WH_heatmaps_for_some_DZ(
    df_clean,
    dz_col="DZ_mm",
    w_col="W",
    h_col="H",
    err_col="Err_mean_mm",
    num_dz_to_plot=4,
    save_dir=None,
):
    """
    몇 개의 대표적인 DZ 값에 대해 W-H vs Error heatmap을 그린다.

    - 각 DZ에 대해:
      x축: W, y축: H, 색: Error (작을수록 좋은 영역)
    - num_dz_to_plot: 최대 몇 개의 DZ slice를 그릴지 (unique 값이 더 적으면 그만큼만)
    """

    unique_dz = np.sort(df_clean[dz_col].unique())
    if len(unique_dz) == 0:
        print("[WARN] DZ 값이 없습니다.")
        return

    # evenly spaced하게 몇 개만 선택
    if len(unique_dz) <= num_dz_to_plot:
        dz_to_plot = unique_dz
    else:
        indices = np.linspace(0, len(unique_dz) - 1, num_dz_to_plot).astype(int)
        dz_to_plot = unique_dz[indices]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for dz_val in dz_to_plot:
        sub = df_clean[df_clean[dz_col] == dz_val].copy()
        if sub.empty:
            continue

        # 피벗: 행=H, 열=W, 값=Error
        pivot = sub.pivot_table(
            index=h_col,
            columns=w_col,
            values=err_col,
            aggfunc="mean",
        )
        pivot = pivot.sort_index(axis=0)  # H 정렬
        pivot = pivot.sort_index(axis=1)  # W 정렬

        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im, label=err_col)
        plt.xticks(
            np.arange(len(pivot.columns)),
            pivot.columns.astype(str),
            rotation=45,
        )
        plt.yticks(
            np.arange(len(pivot.index)),
            pivot.index.astype(str),
        )
        plt.xlabel(w_col)
        plt.ylabel(h_col)
        plt.title(f"DZ={dz_val} at W-H vs Error")

        plt.tight_layout()

        if save_dir is not None:
            fname = os.path.join(save_dir, f"WH_heatmap_DZ_{dz_val}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"[INFO] DZ={dz_val} heatmap을 '{fname}'에 저장했습니다.")
            plt.close()
        else:
            plt.show()


# ===============================
# 메인 예시
# ===============================

if __name__ == "__main__":
    # 1) gp_dataset 정리
    df_clean = load_and_clean_gp_dataset(
        input_csv="gp_dataset.csv",
        output_csv="gp_dataset_clean.csv",
        schedule_col="Schedule_ID",
        dz_col="DZ_mm",
        w_col="W",
        h_col="H",
    )

    # 2) 전체 스케줄 기준 DZ vs Error 플롯
    plot_dz_vs_error_by_WH(
        df_clean,
        dz_col="DZ_mm",
        w_col="W",
        h_col="H",
        err_col="Err_mean_mm",
        save_path="dz_vs_error_all_schedules.png",
    )

    # 3) 대표 DZ 값들에 대해 W-H heatmap 플롯
    plot_WH_heatmaps_for_some_DZ(
        df_clean,
        dz_col="DZ_mm",
        w_col="W",
        h_col="H",
        err_col="Err_mean_mm",
        num_dz_to_plot=4,
        save_dir="heatmaps_by_DZ",
    )
