"""
AQI Predictor - Phase 3.4: Random Forest + SHAP
=================================================
Tree-based model with SHAP feature importance. Trained per horizon.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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

PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 200, "max_depth": 20},
    {"n_estimators": 300, "max_depth": None},
]
EXPERIMENT_NAME = "random_forest_delta"


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def tune_rf(X_train, y_train_d, X_val, y_val_d):
    best_params, best_rmse = None, float("inf")
    for params in PARAM_GRID:
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train_d)
        rmse = float(np.sqrt(((m.predict(X_val) - y_val_d) ** 2).mean()))
        logger.info(f"    n_est={params['n_estimators']}, depth={str(params['max_depth']):<5} "
                    f"→ val_delta_RMSE={rmse:.3f}")
        if rmse < best_rmse:
            best_params, best_rmse = params, rmse
    logger.info(f"    → best: {best_params}")
    return best_params


def generate_shap_plots(model, X_sample, horizon):
    paths = []
    try:
        import shap
        logger.info(f"  Generating SHAP for {horizon}h...")
        explainer = shap.TreeExplainer(model)
        sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
        shap_values = explainer.shap_values(sample)

        for plot_type, suffix in [("dot", "dot"), ("bar", "bar")]:
            fig = plt.figure(figsize=(10, 8) if plot_type == "dot" else (10, 6))
            shap.summary_plot(shap_values, sample, plot_type=plot_type,
                              show=False, max_display=20)
            plt.title(f"SHAP {plot_type.title()} — RF delta {horizon}h", pad=15)
            plt.tight_layout()
            p = MODELS_DIR / f"shap_rf_{horizon}h_{suffix}.png"
            plt.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
        logger.info(f"  SHAP plots saved: {[p.name for p in paths]}")
    except Exception as e:
        logger.warning(f"  SHAP failed (non-fatal): {e}")
    return paths


def train_one_horizon(data, horizon):
    logger.info(f"\n--- Training Random Forest for {horizon}h horizon ---")

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

    # Median imputation (mostly a no-op now thanks to boundary drop in impute_features)
    imputer = SimpleImputer(strategy="median")
    cols = X_train.columns.tolist()
    X_train_i = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
    X_val_i   = pd.DataFrame(imputer.transform(X_val), columns=cols, index=X_val.index)
    X_test_i  = pd.DataFrame(imputer.transform(X_test), columns=cols, index=X_test.index)
    logger.info(f"  Shapes: train={X_train_i.shape}, val={X_val_i.shape}, test={X_test_i.shape}")

    logger.info("  Tuning hyperparameters:")
    best_params = tune_rf(X_train_i, y_train_d, X_val_i, y_val_d)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)),
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

    print_evaluation_report(f"RF-delta {best_params}", horizon, m_train, m_val, m_test, cat_test)

    m_test_d = compute_metrics(y_test_d, pred_test_d)
    logger.info(f"  Delta-scale test: RMSE={m_test_d['RMSE']:.2f}, "
                f"MAE={m_test_d['MAE']:.2f}, R²={m_test_d['R2']:.3f}")

    rf_model = pipe.named_steps["rf"]
    # Skip SHAP in CI — takes 45+ min and risks timeout
    # Set SKIP_SHAP=true in GitHub Actions env to disable
    skip_shap = os.getenv("SKIP_SHAP", "false").lower() == "true" or                 os.getenv("CI", "false").lower() == "true"
    shap_paths = [] if skip_shap else generate_shap_plots(rf_model, X_train_i, horizon)
    if skip_shap:
        logger.info("  SHAP skipped (CI environment)")

    with mlflow.start_run(run_name=f"rf_{horizon}h"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("horizon_h", horizon)
        mlflow.log_param("n_estimators", best_params["n_estimators"])
        mlflow.log_param("max_depth", str(best_params["max_depth"]))
        mlflow.log_param("target_type", "delta")
        mlflow.log_param("n_features", len(X_train.columns))

        for sn, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
            mlflow.log_metric(f"{sn}_RMSE", m["RMSE"])
            mlflow.log_metric(f"{sn}_MAE", m["MAE"])
            mlflow.log_metric(f"{sn}_R2", m["R2"])
            mlflow.log_metric(f"{sn}_n", m["n_samples"])
        mlflow.log_metric("test_delta_RMSE", m_test_d["RMSE"])
        mlflow.log_metric("test_delta_R2", m_test_d["R2"])

        cat_csv = MODELS_DIR / f"rf_{horizon}h_by_category.csv"
        cat_test.to_csv(cat_csv)
        mlflow.log_artifact(str(cat_csv))
        for p in shap_paths:
            mlflow.log_artifact(str(p))

        mlflow.sklearn.log_model(
            pipe, artifact_path=f"rf_{horizon}h",
            registered_model_name=f"rf_{horizon}h",
        )
        logger.info(f"  ✅ Logged to MLflow: rf_{horizon}h")

    return metrics_summary_row("RandomForest", horizon, m_train, m_val, m_test)


def main():
    init_dagshub()
    logger.info("Loading data...")
    data = get_train_data()

    summaries = [train_one_horizon(data, h) for h in FORECAST_HORIZONS]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MODELS_DIR / "rf_metrics.csv", index=False)

    print("\n" + "=" * 72)
    print("RANDOM FOREST — ALL HORIZONS (absolute-scale metrics)")
    print("=" * 72)
    print(summary_df[["model", "horizon_h", "val_RMSE", "val_R2",
                      "test_RMSE", "test_MAE", "test_R2"]].to_string(index=False))
    print(f"\n✅ DagsHub: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
