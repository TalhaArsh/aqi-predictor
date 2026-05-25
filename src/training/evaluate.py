"""
AQI Predictor - Phase 3.2: Shared Evaluation Module
=====================================================
Metric computation + reporting shared across Ridge, RF, LSTM.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

AQI_BINS = [0, 50, 100, 150, 200, 300, 501]
AQI_LABELS = [
    "Good (0-50)",
    "Moderate (51-100)",
    "Unhealthy-Sensitive (101-150)",
    "Unhealthy (151-200)",
    "Very Unhealthy (201-300)",
    "Hazardous (301+)",
]


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def compute_metrics_by_category(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)})
    df["category"] = pd.cut(df["y_true"], bins=AQI_BINS, labels=AQI_LABELS,
                            right=True, include_lowest=True)
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["sq_err"] = (df["y_pred"] - df["y_true"]) ** 2
    return df.groupby("category", observed=True).agg(
        n=("y_true", "size"),
        mean_aqi=("y_true", "mean"),
        MAE=("abs_err", "mean"),
        RMSE=("sq_err", lambda s: np.sqrt(s.mean())),
    ).round(2)


def print_evaluation_report(
    model_name: str, horizon: int,
    metrics_train: Dict[str, float],
    metrics_val: Dict[str, float],
    metrics_test: Dict[str, float],
    category_breakdown_test: Optional[pd.DataFrame] = None,
) -> None:
    print("\n" + "=" * 70)
    print(f"  {model_name} — {horizon}h forecast")
    print("=" * 70)
    print(f"\n{'Split':<10} {'n':>8} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 50)
    for split_name, m in [("Train", metrics_train),
                          ("Val", metrics_val),
                          ("Test", metrics_test)]:
        print(f"{split_name:<10} {m['n_samples']:>8,} "
              f"{m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['R2']:>10.3f}")

    if category_breakdown_test is not None:
        print(f"\nTest MAE by EPA category:")
        print(category_breakdown_test.to_string())

    gap = metrics_val["RMSE"] - metrics_train["RMSE"]
    print(f"\nVal RMSE − Train RMSE: {gap:+.2f}")
    if abs(gap) < 5:
        print("  → Generalizes well.")
    elif gap > 15:
        print("  → ⚠️ Possible overfitting")
    else:
        print("  → Mild generalization gap, acceptable.")


def metrics_summary_row(model_name, horizon, m_train, m_val, m_test) -> dict:
    return {
        "model": model_name, "horizon_h": horizon,
        "train_RMSE": m_train["RMSE"], "train_MAE": m_train["MAE"], "train_R2": m_train["R2"],
        "val_RMSE": m_val["RMSE"], "val_MAE": m_val["MAE"], "val_R2": m_val["R2"],
        "test_RMSE": m_test["RMSE"], "test_MAE": m_test["MAE"], "test_R2": m_test["R2"],
        "n_train": m_train["n_samples"], "n_val": m_val["n_samples"], "n_test": m_test["n_samples"],
    }
