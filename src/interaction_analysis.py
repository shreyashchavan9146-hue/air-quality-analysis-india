
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_with_aqi.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "reports" / "interaction_results.csv"


def run_interaction_analysis():

    df = pd.read_csv(DATA_PATH)

    required = ["AQI", "no2", "rspm", "spm"]

    df[required] = df[required].apply(pd.to_numeric, errors="coerce")

    data = df[required].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    formulas = [
        "AQI ~ no2 + rspm + spm",
        "AQI ~ no2 + rspm",
        "AQI ~ no2 + spm",
        "AQI ~ spm + rspm",
        "AQI ~ spm",
        "AQI ~ no2",
        "AQI ~ rspm",
        "AQI ~ spm + rspm + spm:rspm",
        "AQI ~ no2 + spm + no2:spm"
    ]

    results = []

    for formula in formulas:

        model = smf.ols(formula=formula, data=data).fit()

        results.append({
            "Formula": formula,
            "R2": model.rsquared,
            "Adjusted_R2": model.rsquared_adj,
            "AIC": model.aic,
            "BIC": model.bic
        })

        print("\nFormula:", formula)
        print("R2:", model.rsquared)
        print("Adjusted R2:", model.rsquared_adj)

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nInteraction analysis completed!")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_interaction_analysis()

