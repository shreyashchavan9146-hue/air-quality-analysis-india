import subprocess
import sys

from src.data_loading import load_data
from src.data_cleaning import clean_data
from src.aqi_calculation import calculate_aqi
from src.state_analysis import state_analysis
from src.regression_analysis import regression_analysis
from src.model_evaluation import evaluate_models


def main():

    print("\n================================")
    print(" AIR QUALITY ANALYSIS OF INDIA ")
    print("================================")

    print("\nStep 1: Loading Dataset")
    load_data()

    print("\nStep 2: Data Cleaning")
    clean_data()

    print("\nStep 3: Exploratory Data Analysis")
    subprocess.run(
        [sys.executable, "src/eda.py"], check=True
    )

    print("\nStep 4: AQI Calculation")
    calculate_aqi()

    print("\nStep 5: State-wise Analysis")
    state_analysis()

    print("\nStep 6: Regression Analysis")
    regression_analysis()

    print("\nStep 7: Model Evaluation")
    evaluate_models()

    extra_scripts = [
        "src/vif_analysis.py",
        "src/regression_analysis.py",
        "src/interaction_analysis.py",
        "src/regularization.py",
        "src/model_comparison.py"
    ]

    for script in extra_scripts:
        print(f"\nRunning {script}")
    subprocess.run(
        [sys.executable, script],
            check=True
        )

    # Step 13: AQI Model Prediction
    print("\nStep 13: AQI Model Prediction")
    subprocess.run(
        [sys.executable, "src/model_prediction.py"],
        check=True
    )

    print("\n================================")
    print(" PROJECT EXECUTION COMPLETED ")
    print("================================")

if __name__ == "__main__":
    main()