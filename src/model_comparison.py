
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import mean_squared_error, r2_score


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "air_quality_with_aqi.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "aqi_model_comparison.csv"
)

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def compare_models():

    print("\n================================")
    print(" AQI MODEL COMPARISON")
    print("================================")

    # Step 1: Load dataset
    df = pd.read_csv(DATA_PATH)

    # Step 2: Select features and target
    features = ["so2", "no2", "spm"]
    target = "AQI"

    data = df[features + [target]].copy()

    # Step 3: Convert columns into numeric values
    for column in features + [target]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    X = data[features]
    y = data[target]

    if len(data) < 10:
        raise ValueError(
            "Not enough valid records to train the models."
        )

    # Step 4: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Step 5: Define models
    models = {

        "Linear Regression": LinearRegression(),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=1.0,
                max_iter=10000
            ))
        ])
    }

    # Step 6: Create output folder
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_files = {
        "Linear Regression":
            "linear_regression_aqi_model.pkl",

        "Ridge Regression":
            "ridge_aqi_model.pkl",

        "Lasso Regression":
            "lasso_aqi_model.pkl"
    }

    # Step 7: Train, evaluate and save models
    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "R2_Score": r2
        })

        # Save trained model
        model_path = MODEL_DIR / model_files[name]

        joblib.dump(model, model_path)

        print("RMSE:", rmse)
        print("R2 Score:", r2)
        print("Model saved:", model_path)

    # Step 8: Save comparison results
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Step 9: Display final results
    print("\n================================")
    print(" MODEL COMPARISON RESULTS")
    print("================================")

    print(results_df.to_string(index=False))

    print("\nAll three models saved successfully!")

    print("\nResults saved at:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    compare_models()
