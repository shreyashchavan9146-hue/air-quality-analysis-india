
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_with_aqi.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "ridge_alpha_comparison.csv"


def run_regularization():

    df = pd.read_csv(DATA_PATH)

    features = ["si", "ni", "rpi", "spi"]
    target = "AQI"

    data = df[features + [target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[target], errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    alpha_values = [0.05, 0.1, 1.0, 10.0]
    results = []

    for alpha in alpha_values:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

        model.fit(X_train, y_train)
        MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "ridge_aqi_model.pkl"

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        print("Model saved successfully:", MODEL_PATH)

        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        cv_scores = cross_val_score(
            model, X, y, cv=10, scoring="r2"
        )

        results.append({
            "Alpha": alpha,
            "RMSE": rmse,
            "R2": r2,
            "Mean_CV_R2": cv_scores.mean()
        })

        print("\nAlpha:", alpha)
        print("RMSE:", rmse)
        print("R2 Score:", r2)
        print("Mean CV R2:", cv_scores.mean())

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nRidge regularization completed!")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_regularization()
