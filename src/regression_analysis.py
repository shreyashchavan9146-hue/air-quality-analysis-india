
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_air_quality.csv"

GRAPH_PATH = PROJECT_ROOT / "outputs" / "graphs"
GRAPH_PATH.mkdir(parents=True, exist_ok=True)


def regression_analysis():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Select available pollutant columns
    features = ["so2", "no2", "rspm"]
    target = "spm"

    # Check required columns
    required = features + [target]

    if not all(col in df.columns for col in required):
        print("Required pollutant columns are missing!")
        print("Available columns:", df.columns.tolist())
        return

    # Prepare data
    
    # Prepare data
    df[required] = df[required].apply(pd.to_numeric, errors="coerce")

    data = df[required].dropna()

    X = data[features]
    y = data[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nLinear Regression Results")
    print("-------------------------")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    print("\nModel Coefficients:")
    for feature, coefficient in zip(features, model.coef_):
        print(feature, ":", coefficient)

    print("\nIntercept:", model.intercept_)

    # Actual vs predicted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.4)
    plt.xlabel("Actual SPM")
    plt.ylabel("Predicted SPM")
    plt.title("Actual vs Predicted SPM")
    plt.tight_layout()

    plt.savefig(GRAPH_PATH / "actual_vs_predicted_spm.png")
    plt.close()

    print("\nRegression analysis completed successfully!")


if __name__ == "__main__":
    regression_analysis()
