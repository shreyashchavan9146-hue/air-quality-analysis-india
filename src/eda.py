
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_air_quality.csv"

GRAPH_PATH = PROJECT_ROOT / "outputs" / "graphs"
GRAPH_PATH.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Pollutants available in the dataset
pollutants = ["so2", "no2", "rspm", "spm"]

sns.set_theme(style="whitegrid")


# 1. Histograms
for col in pollutants:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col.upper()}")
        plt.xlabel(col.upper())
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(GRAPH_PATH / f"{col}_histogram.png")
        plt.close()


# 2. Boxplots
available_pollutants = [
    col for col in pollutants if col in df.columns
]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[available_pollutants])
plt.title("Air Pollutant Distribution - India")
plt.tight_layout()
plt.savefig(GRAPH_PATH / "pollutant_boxplot.png")
plt.close()


# 3. Correlation heatmap
plt.figure(figsize=(8, 6))
correlation = df[available_pollutants].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Pollutant Correlation Heatmap")
plt.tight_layout()
plt.savefig(GRAPH_PATH / "correlation_heatmap.png")
plt.close()


print("EDA completed successfully!")
print("Graphs saved in:", GRAPH_PATH)
