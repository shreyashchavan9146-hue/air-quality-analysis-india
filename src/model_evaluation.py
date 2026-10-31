
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_air_quality.csv"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def evaluate_models():

    df = pd.read_csv(DATA_PATH)

    features = ["so2", "no2", "rspm"]
    target = "spm"

    required = features + [target]

    if not all(col in df.columns for col in required):
        print("Required columns are missing!")
        return

    data = df[required].dropna()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        ),
        "Lasso Regression": make_pipeline(
            StandardScaler(),
            Lasso(alpha=0.1, max_iter=10000)
        )
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="r2"
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "CV Mean R2": cv_scores.mean()
        })

    results_df = pd.DataFrame(results)

    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(
        OUTPUT_PATH / "model_comparison.csv",
        index=False
    )

    print("\nModel evaluation completed!")
    print("Results saved in:", OUTPUT_PATH)


if __name__ == "__main__":
    evaluate_models()
