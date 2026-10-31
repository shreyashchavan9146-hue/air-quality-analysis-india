
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_air_quality.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "air_quality_with_aqi.csv"


# SO2 individual index
def calculate_si(so2):
    if so2 <= 40:
        return so2 * (50 / 40)
    elif so2 <= 80:
        return 50 + (so2 - 40) * (50 / 40)
    elif so2 <= 380:
        return 100 + (so2 - 80) * (100 / 300)
    elif so2 <= 800:
        return 200 + (so2 - 380) * (100 / 420)
    elif so2 <= 1600:
        return 300 + (so2 - 800) * (100 / 800)
    else:
        return 400 + (so2 - 1600) * (100 / 800)


# NO2 individual index
def calculate_ni(no2):
    if no2 <= 40:
        return no2 * 50 / 40
    elif no2 <= 80:
        return 50 + (no2 - 40) * 50 / 40
    elif no2 <= 180:
        return 100 + (no2 - 80)
    elif no2 <= 280:
        return 200 + (no2 - 180)
    elif no2 <= 400:
        return 300 + (no2 - 280) * 100 / 120
    else:
        return 400 + (no2 - 400) * 100 / 120


# RSPM individual index
def calculate_rpi(rspm):
    if rspm <= 30:
        return rspm * 50 / 30
    elif rspm <= 60:
        return 50 + (rspm - 30) * 50 / 30
    elif rspm <= 90:
        return 100 + (rspm - 60) * 100 / 30
    elif rspm <= 120:
        return 200 + (rspm - 90) * 100 / 30
    elif rspm <= 250:
        return 300 + (rspm - 120) * 100 / 130
    else:
        return 400 + (rspm - 250) * 100 / 130


# SPM individual index
def calculate_spi(spm):
    if spm <= 50:
        return spm
    elif spm <= 100:
        return 50 + (spm - 50)
    elif spm <= 250:
        return 100 + (spm - 100) * 100 / 150
    elif spm <= 350:
        return 200 + (spm - 250)
    elif spm <= 430:
        return 300 + (spm - 350) * 100 / 80
    else:
        return 400 + (spm - 430) * 100 / 430


def calculate_aqi():

    df = pd.read_csv(DATA_PATH)

    pollutants = ["so2", "no2", "rspm", "spm"]

    for col in pollutants:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate pollutant indices
    df["si"] = df["so2"].apply(calculate_si)
    df["ni"] = df["no2"].apply(calculate_ni)
    df["rpi"] = df["rspm"].apply(calculate_rpi)
    df["spi"] = df["spm"].apply(calculate_spi)

    # Calculate AQI using maximum pollutant index
    df["AQI"] = df[["si", "ni", "rpi", "spi"]].max(axis=1)

    print("\nAQI calculation completed!")

    print("\nFirst five AQI records:")
    print(df[["state", "si", "ni", "rpi", "spi", "AQI"]].head())

    print("\nAQI statistical summary:")
    print(df["AQI"].describe())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFile saved successfully:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    calculate_aqi()

