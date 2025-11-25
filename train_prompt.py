import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DATASET_CSV = "./annotation/dataset.csv"
MODEL_PATH  = "./annotation/model_prompt.pkl"

def main():
    df = pd.read_csv(DATASET_CSV)

    # 특징: DZ, MaxWidth, TopLength
    X = df[["dz_mm", "max_width_mm", "top_length_mm", "point_idx"]].values

    # 타깃: u, v
    y = df[["u", "v"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 파이프라인: 스케일러 + 랜덤포레스트 회귀 (다중 출력)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42
            )
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE (pixel): {rmse:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    main()
