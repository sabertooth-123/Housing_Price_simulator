"""Exploratory data analysis: saves figures to outputs/figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

sns.set_theme(style="whitegrid", palette="viridis")

TARGET = "median_house_value"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[TARGET], bins=60, ax=ax, color="#2a788e")
    ax.axvline(500_001, color="red", ls="--", label="cap at $500k")
    ax.set_title("Median House Value Distribution")
    ax.set_xlabel("Median house value ($)")
    ax.legend()
    save(fig, "target_distribution.png")


def plot_feature_distributions(df):
    features = df.select_dtypes("number").columns.drop(TARGET)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, features):
        sns.histplot(df[col].dropna(), bins=50, ax=ax)
        ax.set_title(col)
    save(fig, "feature_distributions.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix (numeric features)")
    save(fig, "correlation_heatmap.png")


def plot_geographic(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df[TARGET],
                    s=df["population"] / 120, alpha=0.35, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="Median house value ($)")
    ax.set_title("House Values Across California\n(dot size = district population)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    save(fig, "geographic_prices.png")


def plot_ocean_proximity(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = df.groupby("ocean_proximity")[TARGET].median().sort_values().index
    sns.boxplot(data=df, x="ocean_proximity", y=TARGET, order=order, ax=ax)
    ax.set_title("House Value by Ocean Proximity")
    ax.set_xlabel("")
    ax.set_ylabel("Median house value ($)")
    save(fig, "ocean_proximity.png")


def plot_income_vs_value(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sample = df.sample(4000, random_state=42)
    sns.scatterplot(data=sample, x="median_income", y=TARGET, alpha=0.25, ax=ax)
    sns.regplot(data=sample, x="median_income", y=TARGET, scatter=False,
                color="red", ax=ax)
    ax.set_title("Median Income vs House Value (4k sample)")
    ax.set_xlabel("Median income ($10k)")
    ax.set_ylabel("Median house value ($)")
    save(fig, "income_vs_value.png")


def main():
    df = load_data()
    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_geographic(df)
    plot_ocean_proximity(df)
    plot_income_vs_value(df)

    corr = (df.select_dtypes("number").corr()[TARGET]
            .drop(TARGET).sort_values(key=abs, ascending=False))
    print("\nCorrelation with target:")
    print(corr.round(3).to_string())
    print(f"\nMissing total_bedrooms: {df['total_bedrooms'].isna().sum()}")
    print(f"Districts at the $500k cap: {(df[TARGET] >= 500_001).mean():.1%}")


if __name__ == "__main__":
    main()
