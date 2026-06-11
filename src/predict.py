"""Price a district with the saved model.

Usage:
    python src/predict.py    # demo with three sample districts
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

RAW_COLUMNS = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income",
               "ocean_proximity"]

SAMPLES = pd.DataFrame(
    [
        [-122.42, 37.77, 25, 3000, 450, 1200, 460, 8.3, "NEAR BAY"],   # affluent SF
        [-119.77, 36.75, 30, 2400, 470, 1400, 450, 3.5, "INLAND"],     # median Fresno
        [-118.24, 34.05, 40, 2200, 560, 2500, 590, 2.0, "<1H OCEAN"],  # dense LA
    ],
    columns=RAW_COLUMNS,
)


def predict(df: pd.DataFrame) -> pd.Series:
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    with open(MODEL_DIR / "preprocessing.json") as f:
        stats = json.load(f)
    features = engineer_features(df, bedrooms_median=stats["bedrooms_median"])
    return pd.Series(model.predict(features), index=df.index)


if __name__ == "__main__":
    print("Pricing three sample districts:\n")
    preds = predict(SAMPLES)
    for i, price in preds.items():
        row = SAMPLES.loc[i]
        print(f"District at ({row.latitude:.2f}, {row.longitude:.2f}), "
              f"{row.ocean_proximity}, income ${row.median_income * 10_000:,.0f}: "
              f"predicted median value ${price:,.0f}")
