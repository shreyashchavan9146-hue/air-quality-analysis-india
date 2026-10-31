
import pandas as pd
from pathlib import Path

# File paths
DATA_PATH = Path(r"C:\Users\shrey\OneDrive\Desktop\data.csv.csv")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_air_quality.csv"


def clean_data():

    # Load original dataset
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

    print("Original dataset shape:", df.shape)

    # 1. Remove duplicate records
    print("\nDuplicate records:", df.duplicated().sum())
    df = df.drop_duplicates()

    # 2. Check missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # 3. Convert date column if available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 4. Convert pollutant columns to numeric
    pollutant_columns = ["so2", "no2", "rspm", "spm"]

    for col in pollutant_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Fill missing numerical values with median
    for col in pollutant_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 6. Check missing values after cleaning
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # 7. Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset shape:", df.shape)
    print("\nCleaned dataset saved successfully!")
    print("Saved location:", OUTPUT_PATH)


if __name__ == "__main__":
    clean_data()
