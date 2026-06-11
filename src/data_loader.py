"""Load the Kaggle California Housing Prices dataset.

Dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices

"""
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / "housing.csv"

KAGGLE_DATASET = "camnugent/california-housing-prices"


def download_from_kaggle() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )


def load_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        print(f"{RAW_PATH} not found - downloading via Kaggle API...")
        download_from_kaggle()
    return pd.read_csv(RAW_PATH)


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.dtypes}")
    print(f"\nMissing values per column:\n{df.isna().sum()}")
    print(f"\nocean_proximity categories:\n{df['ocean_proximity'].value_counts()}")
    print(f"\nSummary:\n{df.describe().T.round(2)}")
