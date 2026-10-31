
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_with_aqi.csv"

GRAPH_PATH = PROJECT_ROOT / "outputs" / "graphs"
GRAPH_PATH.mkdir(parents=True, exist_ok=True)


def state_analysis():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Check state column
    if "state" not in df.columns:
        print("State column not found!")
        return

    # State-wise average AQI
    state_aqi = (
        df.groupby("state")["AQI"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nTop 10 States with Highest Estimated AQI:")
    print(state_aqi.head(10))

    print("\nTop 10 States with Lowest Estimated AQI:")
    print(state_aqi.tail(10))

    # Plot top 10 states
    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=state_aqi.head(10).values,
        y=state_aqi.head(10).index
    )

    plt.title("Top 10 States by Average Estimated AQI")
    plt.xlabel("Average Estimated AQI")
    plt.ylabel("State")
    plt.tight_layout()

    plt.savefig(GRAPH_PATH / "top_10_states_aqi.png")
    plt.close()

    # Save state-wise results
    output_path = (
        PROJECT_ROOT / "data" / "processed" / "state_wise_aqi.csv"
    )

    state_aqi.to_csv(output_path, header=["Average_AQI"])

    print("\nState-wise analysis completed!")
    print("Results saved:", output_path)


if __name__ == "__main__":
    state_analysis()
