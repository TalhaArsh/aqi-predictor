"""
AQI Predictor - Phase 3.3: Ridge Regression
============================================
Linear baseline with L2 regularization. Trained per horizon on delta
targets; metrics reported on absolute scale (reconstructed at inference).
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
import dagshub

from src.training.data_loader import get_train_data, FORECAST_HORIZONS
from src.training.evaluate import (
    compute_metrics, compute_metrics_by_category,
    print_evaluation_report, metrics_summary_row,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = "aqi-predictor"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
EXPERIMENT_NAME = "ridge_delta"


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def tune_alpha(X_train_s, y_train_delta, X_val_s, y_val_delta):
    best_alpha, best_rmse = None, float("inf")
    for alpha in ALPHAS:
        m = Ridge(alpha=alpha).fit(X_train_s, y_train_delta)
        rmse = float(np.sqrt(((m.predict(X_val_s) - y_val_delta) ** 2).mean()))
        logger.info(f"    alpha={alpha:<6}  val_delta_RMSE={rmse:.3f}")
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    logger.info(f"    → best alpha: {best_alpha}")
    return best_alpha


def train_one_horizon(data, horizon):
    logger.info(f"\n--- Training Ridge for {horizon}h horizon ---")

    X_train = data[f"X_train_{horizon}"]
    X_val   = data[f"X_val_{horizon}"]
    X_test  = data[f"X_test_{horizon}"]
    y_train_d = data[f"y_delta_train_{horizon}"]
    y_val_d   = data[f"y_delta_val_{horizon}"]
    y_test_d  = data[f"y_delta_test_{horizon}"]
    y_train_a = data[f"y_abs_train_{horizon}"]
    y_val_a   = data[f"y_abs_val_{horizon}"]
    y_test_a  = data[f"y_abs_test_{horizon}"]
    aqi_train = data[f"aqi_now_train_{horizon}"]
    aqi_val   = data[f"aqi_now_val_{horizon}"]
    aqi_test  = data[f"aqi_now_test_{horizon}"]

    # Drop rows with any NaN feature (impute_features already removed boundary NaNs)
    mask = X_train.notna().all(axis=1)
    X_train = X_train[mask]; y_train_d = y_train_d[mask]
    y_train_a = y_train_a[mask]; aqi_train = aqi_train[mask]
    mask = X_val.notna().all(axis=1)
    X_val = X_val[mask]; y_val_d = y_val_d[mask]
    y_val_a = y_val_a[mask]; aqi_val = aqi_val[mask]
    mask = X_test.notna().all(axis=1)
    X_test = X_test[mask]; y_test_d = y_test_d[mask]
    y_test_a = y_test_a[mask]; aqi_test = aqi_test[mask]
    logger.info(f"  Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Tune alpha on scaled data
    scaler = StandardScaler().fit(X_train)
    logger.info("  Tuning alpha:")
    best_alpha = tune_alpha(scaler.transform(X_train), y_train_d,
                            scaler.transform(X_val), y_val_d)

    # Final pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha)),
    ])
    pipe.fit(X_train, y_train_d)

    pred_train_d = pipe.predict(X_train)
    pred_val_d   = pipe.predict(X_val)
    pred_test_d  = pipe.predict(X_test)

    pred_train_a = (aqi_train.values + pred_train_d).clip(0, 500)
    pred_val_a   = (aqi_val.values + pred_val_d).clip(0, 500)
    pred_test_a  = (aqi_test.values + pred_test_d).clip(0, 500)

    m_train = compute_metrics(y_train_a, pred_train_a)
    m_val   = compute_metrics(y_val_a, pred_val_a)
    m_test  = compute_metrics(y_test_a, pred_test_a)
    cat_test = compute_metrics_by_category(y_test_a, pred_test_a)

    print_evaluation_report(f"Ridge-delta α={best_alpha}", horizon,
                            m_train, m_val, m_test, cat_test)

    m_test_d = compute_metrics(y_test_d, pred_test_d)
    logger.info(f"  Delta-scale test: RMSE={m_test_d['RMSE']:.2f}, "
                f"MAE={m_test_d['MAE']:.2f}, R²={m_test_d['R2']:.3f}")

    with mlflow.start_run(run_name=f"ridge_{horizon}h"):
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("horizon_h", horizon)
        mlflow.log_param("alpha", best_alpha)
        mlflow.log_param("target_type", "delta")
        mlflow.log_param("n_features", len(X_train.columns))

        for sn, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
            mlflow.log_metric(f"{sn}_RMSE", m["RMSE"])
            mlflow.log_metric(f"{sn}_MAE", m["MAE"])
            mlflow.log_metric(f"{sn}_R2", m["R2"])
            mlflow.log_metric(f"{sn}_n", m["n_samples"])
        mlflow.log_metric("test_delta_RMSE", m_test_d["RMSE"])
        mlflow.log_metric("test_delta_R2", m_test_d["R2"])

        cat_csv = MODELS_DIR / f"ridge_{horizon}h_by_category.csv"
        cat_test.to_csv(cat_csv)
        mlflow.log_artifact(str(cat_csv))

        mlflow.sklearn.log_model(
            pipe, artifact_path=f"ridge_{horizon}h",
            registered_model_name=f"ridge_{horizon}h",
        )
        logger.info(f"  ✅ Logged to MLflow: ridge_{horizon}h")

    return metrics_summary_row(f"Ridge(α={best_alpha})", horizon, m_train, m_val, m_test)


def main():
    init_dagshub()
    logger.info("Loading data...")
    data = get_train_data()

    summaries = [train_one_horizon(data, h) for h in FORECAST_HORIZONS]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MODELS_DIR / "ridge_metrics.csv", index=False)

    print("\n" + "=" * 72)
    print("RIDGE — ALL HORIZONS (absolute-scale metrics)")
    print("=" * 72)
    print(summary_df[["model", "horizon_h", "val_RMSE", "val_R2",
                      "test_RMSE", "test_MAE", "test_R2"]].to_string(index=False))
    print(f"\n✅ DagsHub: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
