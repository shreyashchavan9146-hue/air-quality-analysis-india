
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def predict_aqi():

    print("\n================================")
    print(" AIR QUALITY PREDICTION")
    print("================================")

    # Select model
    print("\nChoose a model:")
    print("1. Linear Regression")
    print("2. Ridge Regression")
    print("3. Lasso Regression")

    choice = input("\nEnter your choice (1-3): ")

    model_files = {
        "1": "linear_regression_aqi_model.pkl",
        "2": "ridge_aqi_model.pkl",
        "3": "lasso_aqi_model.pkl"
    }

    if choice not in model_files:
        print("Invalid choice!")
        return

    model_path = MODEL_DIR / model_files[choice]

    if not model_path.exists():
        print("Model file not found:", model_path)
        return

    # Load trained model
    model = joblib.load(model_path)

    # Get pollutant values
    try:
        so2 = float(input("Enter SO2 value: "))
        no2 = float(input("Enter NO2 value: "))
        spm = float(input("Enter SPM value: "))
    except ValueError:
        print("Please enter valid numeric values.")
        return

    # Prepare input data
    input_data = pd.DataFrame(
        [[so2, no2, spm]],
        columns=["so2", "no2", "spm"]
    )

    # Predict AQI
    prediction = model.predict(input_data)

    print("\n================================")
    print(" PREDICTION RESULT")
    print("================================")

    print("Estimated AQI:", round(float(prediction[0]), 2))


if __name__ == "__main__":
    predict_aqi()
