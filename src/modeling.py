"""Train and compare regression models; save the best one with evaluation figures."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import make_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

sns.set_theme(style="whitegrid")

MODELS = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    ),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.08, max_depth=None,
        l2_regularization=1.0, random_state=42,
    ),
}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "RMSE": root_mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }, pred


def plot_comparison(results: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        order = results.sort_values(metric, ascending=(metric != "R2"))
        fmt = "%.3f" if metric == "R2" else "%.0f"
        bars = ax.barh(order.index, order[metric], color="#2a788e")
        ax.bar_label(bars, fmt=fmt, padding=3)
        ax.set_title(f"Test {metric}" + ("" if metric == "R2" else " ($)"))
        ax.margins(x=0.18)
    fig.suptitle("Model Comparison (test set)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(y_test, pred, name):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].scatter(y_test, pred, alpha=0.2, s=8, color="#440154")
    lims = [0, 520_000]
    axes[0].plot(lims, lims, "r--", lw=1.5)
    axes[0].set_xlabel("Actual ($)")
    axes[0].set_ylabel("Predicted ($)")
    axes[0].set_title(f"{name}: Predicted vs Actual")

    residuals = y_test - pred
    axes[1].scatter(pred, residuals, alpha=0.2, s=8, color="#2a788e")
    axes[1].axhline(0, color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("Predicted ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_model_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_importance(model, X_test, y_test):
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(model, X_test, y_test, n_repeats=5,
                                 random_state=42, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.array(X_test.columns)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#2a788e")
    ax.set_title("Permutation Feature Importance (best model, test set)")
    ax.set_xlabel("Mean decrease in R²")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    X_train, X_test, y_train, y_test, bedrooms_median = make_splits()
    rows = {}
    fitted = {}

    for name, model in MODELS.items():
        cv = cross_val_score(model, X_train, y_train, cv=5,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
        model.fit(X_train, y_train)
        metrics, _ = evaluate(model, X_test, y_test)
        metrics["CV RMSE"] = -cv.mean()
        rows[name] = metrics
        fitted[name] = model
        print(f"{name:<20} CV RMSE={-cv.mean():,.0f}  "
              f"test RMSE={metrics['RMSE']:,.0f}  R2={metrics['R2']:.4f}")

    results = pd.DataFrame(rows).T
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "model_metrics.csv")
    plot_comparison(results)

    best_name = results["RMSE"].idxmin()
    best = fitted[best_name]
    _, pred = evaluate(best, X_test, y_test)
    plot_predictions(y_test, pred, best_name)
    plot_importance(best, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "best_model.pkl")
    # Persist training-set stats needed to rebuild features at predict time
    with open(MODEL_DIR / "preprocessing.json", "w") as f:
        json.dump({"bedrooms_median": bedrooms_median, "model": best_name}, f)

    print(f"\nBest model: {best_name}")
    print("Saved to outputs/models/best_model.pkl")
    print(f"Average prediction error: ${results.loc[best_name, 'MAE']:,.0f}")


if __name__ == "__main__":
    main()
