"""
AQI Predictor - Phase 3.6: CatBoost Multi-Horizon Forecaster
===============================================================
Gradient boosting via CatBoost with everything tuned for noisy
long-horizon AQI prediction.

Why CatBoost (not LightGBM):
- Ordered boosting prevents target leakage in temporal data
- Native categorical features (hour, day_of_week, month, is_weekend)
  → splits on these natively instead of treating as ordinal floats
- Robust default regularization

Long-horizon optimization strategy:
- Per-horizon hyperparameter grids (heavier reg + slower LR for 24/48/72)
- Recency-weighted training samples (recent rows weigh more)
- Early stopping on val to halt before overfit
- SHAP + native feature importance logged for interpretability

Outputs:
- models/catboost_metrics.csv
- models/shap_catboost_{h}h_bar.png  (one per horizon)
- MLflow experiment "catboost_delta", models registered as catboost_{1h..72h}
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
from catboost import CatBoostRegressor, Pool

import mlflow
import mlflow.catboost
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

EXPERIMENT_NAME = "catboost_delta"
CATEGORICAL_FEATURES = ["hour", "day", "month", "day_of_week", "is_weekend"]
SEED = 42
np.random.seed(SEED)


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def recency_weights(n: int, half_life_frac: float = 0.5) -> np.ndarray:
    """Exponential weights: oldest sample ≈ 0.5x weight of newest.
    More weight on recent samples since test set is most recent."""
    ages = np.arange(n)[::-1]  # 0 = newest, n-1 = oldest
    decay_rate = np.log(2) / (n * half_life_frac)
    weights = np.exp(-decay_rate * ages)
    return weights / weights.mean()  # normalize so mean weight = 1


def get_search_grid(horizon: int) -> List[dict]:
    """Per-horizon hyperparameter grid.

    Short horizons: standard config (fast convergence, less reg).
    Long horizons (24h, 48h, 72h): heavier regularization, slower LR,
    deeper trees + bagging to combat seasonal distribution shift.
    """
    if horizon <= 6:
        return [
            {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3,
             "iterations": 1500, "bagging_temperature": 0.0},
            {"learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 3,
             "iterations": 2000, "bagging_temperature": 0.5},
        ]
    elif horizon <= 12:
        return [
            {"learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5,
             "iterations": 2000, "bagging_temperature": 0.5},
            {"learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 5,
             "iterations": 2500, "bagging_temperature": 1.0},
        ]
    else:
        # Long horizons: extra search budget
        return [
            {"learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 7,
             "iterations": 2500, "bagging_temperature": 1.0},
            {"learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 7,
             "iterations": 2500, "bagging_temperature": 1.0},
            {"learning_rate": 0.01, "depth": 6, "l2_leaf_reg": 10,
             "iterations": 3500, "bagging_temperature": 1.0},
            {"learning_rate": 0.01, "depth": 8, "l2_leaf_reg": 10,
             "iterations": 3500, "bagging_temperature": 2.0},
            {"learning_rate": 0.01, "depth": 10, "l2_leaf_reg": 15,
             "iterations": 4000, "bagging_temperature": 2.0},
        ]


def get_cat_feature_indices(feature_names: List[str]) -> List[int]:
    return [i for i, name in enumerate(feature_names) if name in CATEGORICAL_FEATURES]


def generate_shap_plots(model, X_val_df, h, max_samples=500):
    """Generate SHAP bar plot for the model on a sample of val data."""
    try:
        sample = X_val_df.sample(min(max_samples, len(X_val_df)), random_state=SEED)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=20)
        bar_path = MODELS_DIR / f"shap_catboost_{h}h_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"  SHAP bar saved: {bar_path.name}")
        return [str(bar_path)]
    except Exception as e:
        logger.warning(f"  SHAP failed for {h}h: {e}")
        return []


def train_catboost_for_horizon(data, h: int, feature_names: List[str]):
    logger.info(f"\n--- Training CatBoost for {h}h horizon ---")

    X_train = data[f"X_train_{h}"].copy()
    y_train_d = data[f"y_delta_train_{h}"].copy()
    X_val = data[f"X_val_{h}"].copy()
    y_val_d = data[f"y_delta_val_{h}"].copy()
    X_test = data[f"X_test_{h}"].copy()
    y_test_d = data[f"y_delta_test_{h}"].copy()

    aqi_now_train = data[f"aqi_now_train_{h}"]
    aqi_now_val = data[f"aqi_now_val_{h}"]
    aqi_now_test = data[f"aqi_now_test_{h}"]
    y_test_a = data[f"y_abs_test_{h}"]
    y_val_a = data[f"y_abs_val_{h}"]
    y_train_a = data[f"y_abs_train_{h}"]

    logger.info(f"  Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    cat_idx = get_cat_feature_indices(feature_names)

    # Cast categorical columns to int
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    # Recency weights — more weight on recent training samples
    sample_weights = recency_weights(len(X_train), half_life_frac=0.4)
    logger.info(f"  Sample weights: oldest={sample_weights[0]:.3f}, "
                f"newest={sample_weights[-1]:.3f}")

    grid = get_search_grid(h)
    logger.info(f"  Tuning {len(grid)} configurations:")

    best_val_rmse = float("inf")
    best_model = None
    best_params = None

    for params in grid:
        model = CatBoostRegressor(
            **params,
            loss_function="RMSE",
            random_seed=SEED,
            verbose=False,
            cat_features=cat_idx,
            early_stopping_rounds=75,
        )
        train_pool = Pool(X_train, label=y_train_d, cat_features=cat_idx,
                          weight=sample_weights)
        val_pool = Pool(X_val, label=y_val_d, cat_features=cat_idx)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

        pred_val_d = model.predict(X_val)
        val_rmse_delta = float(np.sqrt(np.mean((pred_val_d - y_val_d.values) ** 2)))
        n_trees = model.tree_count_

        logger.info(f"    lr={params['learning_rate']}, depth={params['depth']}, "
                    f"l2={params['l2_leaf_reg']}, bag={params['bagging_temperature']}, "
                    f"trees={n_trees:<5} → val_d_RMSE={val_rmse_delta:.3f}")

        if val_rmse_delta < best_val_rmse:
            best_val_rmse = val_rmse_delta
            best_params = {**params, "n_trees_used": n_trees}
            best_model = model

    logger.info(f"    → best: lr={best_params['learning_rate']}, "
                f"depth={best_params['depth']}, l2={best_params['l2_leaf_reg']}, "
                f"trees={best_params['n_trees_used']}")

    # Predictions
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

    print_evaluation_report(f"CatBoost-delta", h, m_train, m_val, m_test, cat_metrics)

    m_test_delta = compute_metrics(y_test_d.values, pred_test_d)
    logger.info(f"  Delta-scale test: RMSE={m_test_delta['RMSE']:.2f}, "
                f"R²={m_test_delta['R2']:.3f}")

    # SHAP only for medium-to-long horizons (most interpretation value)
    shap_paths = []
    if h in [6, 24, 48, 72]:
        logger.info(f"  Generating SHAP for {h}h...")
        shap_paths = generate_shap_plots(best_model, X_val, h)

    # Log to MLflow
    with mlflow.start_run(run_name=f"catboost_{h}h"):
        mlflow.log_param("horizon_h", h)
        mlflow.log_param("model_type", "CatBoost")
        mlflow.log_param("target_type", "delta")
        mlflow.log_param("sample_weighting", "recency_exp_half_life_0.4")
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("categorical_features", str(CATEGORICAL_FEATURES))

        for split_name, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
            mlflow.log_metric(f"{split_name}_RMSE", m["RMSE"])
            mlflow.log_metric(f"{split_name}_MAE", m["MAE"])
            mlflow.log_metric(f"{split_name}_R2", m["R2"])
            mlflow.log_metric(f"{split_name}_n_samples", m["n_samples"])
        mlflow.log_metric("test_RMSE_delta", m_test_delta["RMSE"])
        mlflow.log_metric("test_R2_delta", m_test_delta["R2"])

        cat_path = MODELS_DIR / f"catboost_{h}h_by_category.csv"
        cat_metrics.to_csv(cat_path)
        mlflow.log_artifact(str(cat_path))

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        fi_path = MODELS_DIR / f"catboost_{h}h_feature_importance.csv"
        feat_imp.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        for shap_path in shap_paths:
            mlflow.log_artifact(shap_path)

        mlflow.catboost.log_model(
            best_model,
            artifact_path=f"catboost_{h}h",
            registered_model_name=f"catboost_{h}h",
        )
        logger.info(f"  ✅ Logged to MLflow: catboost_{h}h")

    return {
        "model": "CatBoost",
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
            row = train_catboost_for_horizon(data, h, feature_names)
            summaries.append(row)
        except Exception as e:
            logger.error(f"Failed for {h}h: {e}", exc_info=True)
            continue

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MODELS_DIR / "catboost_metrics.csv", index=False)

    print("\n" + "=" * 72)
    print("CATBOOST — ALL HORIZONS (absolute-scale metrics)")
    print("=" * 72)
    print(summary_df[["model", "horizon_h", "val_RMSE", "val_R2",
                      "test_RMSE", "test_MAE", "test_R2"]].to_string(index=False))
    print(f"\n✅ DagsHub: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
