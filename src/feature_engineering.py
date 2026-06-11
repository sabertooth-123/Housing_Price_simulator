"""Feature engineering and train/test split for the Kaggle housing dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "median_house_value"

# Fixed category list so one-hot columns are deterministic at predict time
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# Major metro centers (lat, lon) for distance features
CITIES = {
    "dist_to_la": (34.05, -118.24),
    "dist_to_sf": (37.77, -122.42),
    "dist_to_sd": (32.72, -117.16),
    "dist_to_sacramento": (38.58, -121.49),
}


def engineer_features(df: pd.DataFrame, bedrooms_median: float | None = None) -> pd.DataFrame:
    """Return a model-ready numeric dataframe (target column kept if present).

    `bedrooms_median` is the imputation value learned from training data;
    pass it explicitly at predict time to avoid leaking statistics.
    """
    df = df.copy()

    if bedrooms_median is None:
        bedrooms_median = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(bedrooms_median)

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df["log_population"] = np.log1p(df["population"])

    for name, (lat, lon) in CITIES.items():
        df[name] = np.hypot(df["latitude"] - lat, df["longitude"] - lon)
    df["dist_to_nearest_city"] = df[list(CITIES)].min(axis=1)

    ocean = pd.Categorical(df["ocean_proximity"], categories=OCEAN_CATEGORIES)
    dummies = pd.get_dummies(ocean, prefix="ocean").set_axis(df.index)
    df = pd.concat([df.drop(columns="ocean_proximity"), dummies], axis=1)

    return df


def make_splits(test_size: float = 0.2, random_state: int = 42):
    """Engineer features and return X_train, X_test, y_train, y_test.

    Stratified on income quintiles so both splits cover the income range.
    The bedrooms imputation median is computed on the training rows only.
    """
    raw = load_data()
    income_bins = pd.qcut(raw["median_income"], q=5, labels=False)
    train_raw, test_raw = train_test_split(
        raw, test_size=test_size, random_state=random_state, stratify=income_bins
    )

    bedrooms_median = train_raw["total_bedrooms"].median()
    train_df = engineer_features(train_raw, bedrooms_median)
    test_df = engineer_features(test_raw, bedrooms_median)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    X_train = train_df.drop(columns=TARGET)
    X_test = test_df.drop(columns=TARGET)
    return X_train, X_test, train_df[TARGET], test_df[TARGET], bedrooms_median


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, med = make_splits()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Bedrooms imputation median (train only): {med}")
    print(f"Features: {list(X_train.columns)}")
    full = pd.concat([X_train, y_train], axis=1)
    corr = full.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\nCorrelation with target (engineered features included):")
    print(corr.round(3).to_string())
