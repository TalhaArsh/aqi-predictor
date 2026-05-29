"""
AQI Predictor - Phase 3.7: XGBoost Multi-Horizon Forecaster
==============================================================
Gradient boosting via XGBoost — fastest tree-based regressor, with
histogram-based tree construction and strong regularization options.

Differences from CatBoost run:
- XGBoost doesn't natively handle categoricals as well, so we use
  enable_categorical=True with pandas category dtype
- tree_method='hist' for speed + low memory
- Built-in subsample + colsample for stochastic gradient boosting
- Different default loss landscape → diversity in our ensemble

Strategy matches CatBoost:
- Per-horizon hyperparameter grids (heavier reg for 24/48/72)
- Recency-weighted training samples
- Early stopping on val set
- SHAP + native feature importance

Outputs:
- models/xgboost_metrics.csv
- models/shap_xgboost_{h}h_bar.png  (one per horizon)
- MLflow experiment "xgboost_delta", models registered as xgboost_{1h..72h}
"""

import os
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import shap
import xgboost as xgb

import mlflow
import mlflow.xgboost
import dagshub

from src.training.data_loader import get_train_data, FORECAST_HORIZONS
from src.training.evaluate import (
    compute_metrics, compute_metrics_by_category,
    print_evaluation_report,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = "aqi-predictor"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "xgboost_delta"
CATEGORICAL_FEATURES = ["hour", "day", "month", "day_of_week", "is_weekend"]
SEED = 42
np.random.seed(SEED)


def init_dagshub():
    # Authenticate with token (prevents interactive OAuth in CI)
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def recency_weights(n: int, half_life_frac: float = 0.4) -> np.ndarray:
    ages = np.arange(n)[::-1]
    decay_rate = np.log(2) / (n * half_life_frac)
    weights = np.exp(-decay_rate * ages)
    return weights / weights.mean()


def get_search_grid(horizon: int) -> List[dict]:
    """Per-horizon hyperparameter grid for XGBoost.

    Long horizons get heavier regularization (reg_alpha, reg_lambda),
    slower learning rate, more trees, and stochastic sampling.
    """
    common = {
        "tree_method": "hist",
        "enable_categorical": True,
        "objective": "reg:squarederror",
    }
    if horizon <= 6:
        return [
            {**common, "learning_rate": 0.05, "max_depth": 6, "n_estimators": 1500,
             "subsample": 0.9, "colsample_bytree": 0.9,
             "reg_alpha": 0.0, "reg_lambda": 1.0, "min_child_weight": 1},
            {**common, "learning_rate": 0.03, "max_depth": 8, "n_estimators": 2000,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3},
        ]
    elif horizon <= 12:
        return [
            {**common, "learning_rate": 0.03, "max_depth": 6, "n_estimators": 2000,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_alpha": 0.1, "reg_lambda": 2.0, "min_child_weight": 5},
            {**common, "learning_rate": 0.02, "max_depth": 8, "n_estimators": 2500,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_alpha": 0.1, "reg_lambda": 3.0, "min_child_weight": 5},
        ]
    else:
        # Long horizons: more configs + heavier regularization
        return [
            {**common, "learning_rate": 0.02, "max_depth": 6, "n_estimators": 2500,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_alpha": 0.3, "reg_lambda": 5.0, "min_child_weight": 10},
            {**common, "learning_rate": 0.02, "max_depth": 8, "n_estimators": 2500,
             "subsample": 0.7, "colsample_bytree": 0.7,
             "reg_alpha": 0.3, "reg_lambda": 5.0, "min_child_weight": 10},
            {**common, "learning_rate": 0.01, "max_depth": 6, "n_estimators": 4000,
             "subsample": 0.7, "colsample_bytree": 0.7,
             "reg_alpha": 0.5, "reg_lambda": 7.0, "min_child_weight": 15},
            {**common, "learning_rate": 0.01, "max_depth": 8, "n_estimators": 4000,
             "subsample": 0.7, "colsample_bytree": 0.7,
             "reg_alpha": 0.5, "reg_lambda": 10.0, "min_child_weight": 20},
            {**common, "learning_rate": 0.01, "max_depth": 4, "n_estimators": 5000,
             "subsample": 0.6, "colsample_bytree": 0.6,
             "reg_alpha": 1.0, "reg_lambda": 15.0, "min_child_weight": 25},
        ]


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical feature columns to pandas category dtype for XGBoost."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def generate_shap_plots(model, X_val_df, h, max_samples=500):
    try:
        sample = X_val_df.sample(min(max_samples, len(X_val_df)), random_state=SEED)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=20)
        bar_path = MODELS_DIR / f"shap_xgboost_{h}h_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"  SHAP bar saved: {bar_path.name}")
        return [str(bar_path)]
    except Exception as e:
        logger.warning(f"  SHAP failed for {h}h: {e}")
        return []


def train_xgb_for_horizon(data, h: int, feature_names: List[str]):
    logger.info(f"\n--- Training XGBoost for {h}h horizon ---")

    X_train = cast_categoricals(data[f"X_train_{h}"])
    y_train_d = data[f"y_delta_train_{h}"]
    X_val = cast_categoricals(data[f"X_val_{h}"])
    y_val_d = data[f"y_delta_val_{h}"]
    X_test = cast_categoricals(data[f"X_test_{h}"])
    y_test_d = data[f"y_delta_test_{h}"]

    aqi_now_train = data[f"aqi_now_train_{h}"]
    aqi_now_val = data[f"aqi_now_val_{h}"]
    aqi_now_test = data[f"aqi_now_test_{h}"]
    y_test_a = data[f"y_abs_test_{h}"]
    y_val_a = data[f"y_abs_val_{h}"]
    y_train_a = data[f"y_abs_train_{h}"]

    logger.info(f"  Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    sample_weights = recency_weights(len(X_train), half_life_frac=0.4)
    logger.info(f"  Sample weights: oldest={sample_weights[0]:.3f}, "
                f"newest={sample_weights[-1]:.3f}")

    grid = get_search_grid(h)
    logger.info(f"  Tuning {len(grid)} configurations:")

    best_val_rmse = float("inf")
    best_model = None
    best_params = None

    for params in grid:
        model = xgb.XGBRegressor(
            **params,
            random_state=SEED,
            early_stopping_rounds=75,
            eval_metric="rmse",
            verbosity=0,
        )
        model.fit(
            X_train, y_train_d,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_d)],
            verbose=False,
        )

        pred_val_d = model.predict(X_val)
        val_rmse_delta = float(np.sqrt(np.mean((pred_val_d - y_val_d.values) ** 2)))
        n_trees_used = model.best_iteration + 1 if model.best_iteration is not None else params["n_estimators"]

        logger.info(f"    lr={params['learning_rate']}, depth={params['max_depth']}, "
                    f"lambda={params['reg_lambda']}, trees={n_trees_used:<5} → "
                    f"val_d_RMSE={val_rmse_delta:.3f}")

        if val_rmse_delta < best_val_rmse:
            best_val_rmse = val_rmse_delta
            best_params = {**params, "n_trees_used": n_trees_used}
            best_model = model

    logger.info(f"    → best: lr={best_params['learning_rate']}, "
                f"depth={best_params['max_depth']}, "
                f"lambda={best_params['reg_lambda']}, "
                f"trees={best_params['n_trees_used']}")

    pred_train_d = best_model.predict(X_train)
    pred_val_d = best_model.predict(X_val)
    pred_test_d = best_model.predict(X_test)

    pred_train_abs = np.clip(aqi_now_train.values + pred_train_d, 0, 500)
    pred_val_abs = np.clip(aqi_now_val.values + pred_val_d, 0, 500)
    pred_test_abs = np.clip(aqi_now_test.values + pred_test_d, 0, 500)

    m_train = compute_metrics(y_train_a.values, pred_train_abs)
    m_val = compute_metrics(y_val_a.values, pred_val_abs)
    m_test = compute_metrics(y_test_a.values, pred_test_abs)
    cat_metrics = compute_metrics_by_category(y_test_a.values, pred_test_abs)

    print_evaluation_report(f"XGBoost-delta", h, m_train, m_val, m_test, cat_metrics)

    m_test_delta = compute_metrics(y_test_d.values, pred_test_d)
    logger.info(f"  Delta-scale test: RMSE={m_test_delta['RMSE']:.2f}, "
                f"R²={m_test_delta['R2']:.3f}")

    skip_shap = os.getenv("SKIP_SHAP", "false").lower() == "true" or \
                os.getenv("CI", "false").lower() == "true"
    shap_paths = []
    if not skip_shap and h in [6, 24, 48, 72]:
        logger.info(f"  Generating SHAP for {h}h...")
        shap_paths = generate_shap_plots(best_model, X_val, h)

    with mlflow.start_run(run_name=f"xgboost_{h}h"):
        mlflow.log_param("horizon_h", h)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("target_type", "delta")
        mlflow.log_param("sample_weighting", "recency_exp_half_life_0.4")
        for k, v in best_params.items():
            mlflow.log_param(k, str(v) if not isinstance(v, (int, float, str)) else v)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("categorical_features", str(CATEGORICAL_FEATURES))

        for split_name, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
            mlflow.log_metric(f"{split_name}_RMSE", m["RMSE"])
            mlflow.log_metric(f"{split_name}_MAE", m["MAE"])
            mlflow.log_metric(f"{split_name}_R2", m["R2"])
            mlflow.log_metric(f"{split_name}_n_samples", m["n_samples"])
        mlflow.log_metric("test_RMSE_delta", m_test_delta["RMSE"])
        mlflow.log_metric("test_R2_delta", m_test_delta["R2"])

        cat_path = MODELS_DIR / f"xgboost_{h}h_by_category.csv"
        cat_metrics.to_csv(cat_path)
        mlflow.log_artifact(str(cat_path))

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi_path = MODELS_DIR / f"xgboost_{h}h_feature_importance.csv"
        feat_imp.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        for sp in shap_paths:
            mlflow.log_artifact(sp)

        mlflow.xgboost.log_model(
            best_model,
            artifact_path=f"xgboost_{h}h",
            registered_model_name=f"xgboost_{h}h",
        )
        logger.info(f"  ✅ Logged to MLflow: xgboost_{h}h")

    return {
        "model": "XGBoost",
        "horizon_h": h,
        "best_params": str(best_params),
        "train_RMSE": m_train["RMSE"], "train_R2": m_train["R2"],
        "val_RMSE": m_val["RMSE"], "val_R2": m_val["R2"],
        "test_RMSE": m_test["RMSE"], "test_MAE": m_test["MAE"], "test_R2": m_test["R2"],
        "test_RMSE_delta": m_test_delta["RMSE"], "test_R2_delta": m_test_delta["R2"],
    }


def main():
    init_dagshub()
    logger.info("Loading data...")
    data = get_train_data()
    feature_names = data["feature_names"]

    summaries = []
    for h in FORECAST_HORIZONS:
        try:
            row = train_xgb_for_horizon(data, h, feature_names)
            summaries.append(row)
        except Exception as e:
            logger.error(f"Failed for {h}h: {e}", exc_info=True)
            continue

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MODELS_DIR / "xgboost_metrics.csv", index=False)

    print("\n" + "=" * 72)
    print("XGBOOST — ALL HORIZONS (absolute-scale metrics)")
    print("=" * 72)
    print(summary_df[["model", "horizon_h", "val_RMSE", "val_R2",
                      "test_RMSE", "test_MAE", "test_R2"]].to_string(index=False))
    print(f"\n✅ DagsHub: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
