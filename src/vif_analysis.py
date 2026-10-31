
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_with_aqi.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "vif_analysis.csv"


def calculate_vif(data, features, model_name):

    X = data[features].copy()

    # Convert columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove infinite and missing values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    if X.shape[1] < 2:
        print(f"Not enough valid features for {model_name}")
        return pd.DataFrame()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    vif_data["Model"] = model_name

    return vif_data


def run_vif_analysis():

    df = pd.read_csv(DATA_PATH)

    # Three predictor groups from the notebook
    model_1 = ["si", "ni", "rpi", "spi"]
    model_2 = ["so2", "no2", "spm"]
    model_3 = ["rspm", "spm", "spi", "rpi"]

    all_results = []

    for features, name in [
        (model_1, "Model 1 - Subindices"),
        (model_2, "Model 2 - Pollutants"),
        (model_3, "Model 3 - RSPM and SPM")
    ]:

        results = calculate_vif(df, features, name)

        if not results.empty:
            print(f"\n{name}")
            print(results.to_string(index=False))
            all_results.append(results)

    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_results.to_csv(OUTPUT_PATH, index=False)

        print("\nVIF analysis completed!")
        print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_vif_analysis()

