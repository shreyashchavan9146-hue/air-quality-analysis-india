
import pandas as pd
from pathlib import Path

# Your actual dataset path
DATA_PATH = Path(r"C:\Users\shrey\OneDrive\Desktop\data.csv.csv")


def load_data():

    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

        print("\nDataset loaded successfully!")

        # First 5 rows
        print("\nFirst 5 rows:")
        print(df.head())

        # Dataset shape
        print("\nDataset shape:")
        print(df.shape)

        # Column names
        print("\nColumn names:")
        print(df.columns.tolist())

        # Dataset information
        print("\nDataset information:")
        df.info()

        # Missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Statistical summary
        print("\nStatistical summary:")
        print(df.describe(include="all"))

        return df

    except FileNotFoundError:
        print("Error: CSV file not found!")
        print("Check this path:", DATA_PATH)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    load_data()
