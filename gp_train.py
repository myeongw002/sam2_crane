# gp_train_and_heatmap.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    WhiteKernel
)
from sklearn.model_selection import train_test_split
from joblib import dump


# ---------------------------------------------------------
# 1) 데이터 로드 & 전처리
# ---------------------------------------------------------
def load_gp_dataset(csv_path: str) -> pd.DataFrame:
    """
    gp_dataset_clean.csv 를 읽어서 기본 전처리를 수행한다.
    기대하는 컬럼:
        - DZ_mm
        - W
        - H
        - Err_mean_mm
    다른 이름이면 여기에서 rename 해주면 됨.
    """
    df = pd.read_csv(csv_path)

    # 필요 컬럼만 남기기 (있으면)
    expected_cols = ["DZ_mm", "W", "H", "Err_mean_mm"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"다음 컬럼이 없음: {missing}. CSV 헤더를 확인해 주세요.")

    # 결측치 제거
    df = df.dropna(subset=expected_cols).reset_index(drop=True)

    return df


# ---------------------------------------------------------
# 2) GPR 학습 함수
# ---------------------------------------------------------
def train_gp(df: pd.DataFrame) -> GaussianProcessRegressor:
    """
    입력: df (DZ_mm, W, H, Err_mean_mm)
    출력: 학습된 GaussianProcessRegressor 객체
    """

    # 입력 X: (DZ_mm, W, H)
    X = df[["DZ_mm", "W", "H"]].values.astype(float)
    # 타깃 y: Err_mean_mm
    y = df["Err_mean_mm"].values.astype(float)

    # 간단히 train/val 나눠서 generalization 확인
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 커널 정의
    # - C: 전체 스케일
    # - RBF: (DZ,W,H) 각각에 대한 길이 스케일
    # - WhiteKernel: 관측 노이즈
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[1000.0, 5.0, 5.0],
        length_scale_bounds=(1e-2, 1e4)
    ) + WhiteKernel(
        noise_level=10.0,
        noise_level_bounds=(1e-5, 1e3)
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=0.0,          # WhiteKernel로 노이즈를 처리하므로 alpha=0
        n_restarts_optimizer=5,
        random_state=42
    )

    # 학습
    print("▶ GPR 학습 시작...")
    gpr.fit(X_train, y_train)
    print("▶ 학습 완료")

    # 간단한 검증 MSE 출력
    y_pred, y_std = gpr.predict(X_val, return_std=True)
    mse = np.mean((y_pred - y_val) ** 2)
    print(f"Validation MSE: {mse:.3f}")
    print(f"최종 커널: {gpr.kernel_}")

    # --- Validation Scatter Plot ---
    plot_validation_scatter(
        y_val,
        y_pred,
        out_path="./models/validation_scatter.png"
    )

    return gpr


# ---------------------------------------------------------
# 3) 특정 DZ에서 (W,H) grid에 대한 예측 + 최소 Error 탐색
# ---------------------------------------------------------
def predict_grid_for_dz(
    gpr: GaussianProcessRegressor,
    dz_value: float,
    W_range=(-10, 10),
    H_range=(-10, 10),
    num_W=21,
    num_H=21
):
    """
    특정 DZ에서 W,H grid를 만들어 GPR로 예측.
    """
    Ws = np.linspace(W_range[0], W_range[1], num_W)
    Hs = np.linspace(H_range[0], H_range[1], num_H)

    WW, HH = np.meshgrid(Ws, Hs)  # shape: (num_H, num_W)

    X_grid = np.column_stack([
        np.full(WW.size, dz_value),
        WW.ravel(),
        HH.ravel(),
    ])

    y_pred, y_std = gpr.predict(X_grid, return_std=True)
    y_pred_grid = y_pred.reshape(WW.shape)
    y_std_grid = y_std.reshape(WW.shape)

    # 최소 Error 위치 찾기
    idx_min = np.argmin(y_pred)
    best_W = X_grid[idx_min, 1]
    best_H = X_grid[idx_min, 2]
    best_err = y_pred[idx_min]

    return WW, HH, y_pred_grid, y_std_grid, best_W, best_H, best_err


# ---------------------------------------------------------
# 4) GPR 모델을 파일로 저장
# ---------------------------------------------------------
def save_gp_model(gpr: GaussianProcessRegressor, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump(gpr, out_path)
    print(f"▶ GPR 모델 저장 완료: {out_path}")


# ---------------------------------------------------------
# 5) 각 DZ slice별 Heatmap (실측 기반) 자동 생성
# ---------------------------------------------------------
def plot_empirical_heatmaps_by_dz(
    df: pd.DataFrame,
    out_dir: str = "./heatmaps_empirical",
    min_samples_per_dz: int = 5,
    dz_round: int = 1,
):
    """
    raw 데이터 기반으로 각 DZ slice마다 (W,H) -> Err_mean_mm 평균값 heatmap 생성.
    - DZ_mm 값이 다양한 경우, 'dz_round' 간격으로 반올림해서 그룹핑 가능.
      예: dz_round=10 인 경우, 1743 → 1740 으로 묶임.
    """

    os.makedirs(out_dir, exist_ok=True)

    # DZ를 적당히 라운딩해서 그룹핑 (높이 값이 조금씩 달라도 묶을 수 있게)
    df = df.copy()
    df["DZ_group"] = (df["DZ_mm"] / dz_round).round() * dz_round

    dz_groups = df["DZ_group"].unique()
    dz_groups = np.sort(dz_groups)

    print(f"▶ 발견된 DZ 그룹: {dz_groups}")

    for dz in dz_groups:
        df_dz = df[df["DZ_group"] == dz]

        if len(df_dz) < min_samples_per_dz:
            print(f"  - DZ={dz} (샘플 {len(df_dz)}개) → 스킵")
            continue

        # 피벗테이블: index=H, columns=W, value=Error 평균
        pivot = pd.pivot_table(
            df_dz,
            values="Err_mean_mm",
            index="H",
            columns="W",
            aggfunc="mean"
        )

        H_values = pivot.index.values
        W_values = pivot.columns.values
        Z = pivot.values  # shape (len(H), len(W))

        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                W_values.min(), W_values.max(),
                H_values.min(), H_values.max()
            ]
        )
        plt.colorbar(im, label="Mean Error [mm]")
        plt.xlabel("W 파라미터")
        plt.ylabel("H 파라미터")
        plt.title(f"DZ ≈ {dz} mm 의 (W,H) → Error Heatmap")

        fname = os.path.join(out_dir, f"heatmap_DZ_{int(dz)}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  - 저장 완료: {fname}")


def plot_validation_scatter(y_val, y_pred, out_path=None):
    """
    실측 vs 예측 Scatter Plot
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred, s=40, alpha=0.7, edgecolors='k')

    # y=x 기준선 (예측이 완벽하면 모든 점이 여기에 위치)
    min_v = min(y_val.min(), y_pred.min())
    max_v = max(y_val.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label="Ideal (y=x)")

    plt.xlabel("Actual Error [mm]")
    plt.ylabel("Predicted Error [mm]")
    plt.title("Validation Scatter Plot (Actual vs Predicted)")
    plt.grid(True)
    plt.legend()

    if out_path is not None:
        plt.savefig(out_path, dpi=150)
        print(f"▶ Validation scatter saved: {out_path}")
    plt.show()

# ---------------------------------------------------------
# 6) 메인: 한 번에 실행하는 예시
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 데이터 로드
    csv_path = "./gp_dataset_clean.csv"   # 경로 수정
    df = load_gp_dataset(csv_path)

    # 2. Empirical heatmap (raw 데이터 기반)
    plot_empirical_heatmaps_by_dz(
        df,
        out_dir="./heatmaps_empirical",
        min_samples_per_dz=5,
        dz_round=10,   # DZ 10mm 단위로 그룹핑 (1100, 1700, 3600 등)
    )

    # 3. GPR 학습
    gpr = train_gp(df)

    # 4. GPR 모델 저장 (원하면)
    save_gp_model(gpr, "./models/gp_dz_wh_err.joblib")

    # 5. 예시: DZ=1800 일 때 (W,H) grid 예측 및 최적점 확인
    target_dz = 1800.0
    WW, HH, y_pred_grid, y_std_grid, best_W, best_H, best_err = predict_grid_for_dz(
        gpr, target_dz,
        W_range=(df["W"].min(), df["W"].max()),
        H_range=(df["H"].min(), df["H"].max()),
        num_W=41,
        num_H=41,
    )

    print(f"\n[DZ={target_dz} mm 에서 GPR 기준 최소 Error 예측]")
    print(f"  best W  = {best_W:.3f}")
    print(f"  best H  = {best_H:.3f}")
    print(f"  best Err= {best_err:.3f} mm")

    # GPR 기반 heatmap도 그리고 싶으면 아래 참고
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        y_pred_grid,
        origin="lower",
        aspect="auto",
        extent=[WW.min(), WW.max(), HH.min(), HH.max()]
    )
    plt.colorbar(im, label="Predicted Error [mm]")
    plt.xlabel("W 파라미터")
    plt.ylabel("H 파라미터")
    plt.title(f"GPR 예측 Heatmap (DZ={target_dz} mm)")
    plt.tight_layout()
    plt.savefig(f"./heatmaps_empirical/gpr_heatmap_DZ_{int(target_dz)}.png", dpi=150)
    plt.close()

