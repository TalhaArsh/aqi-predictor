"""
AQI Predictor - Phase 3.7: Permutation Feature Importance
============================================================
Computes permutation importance for Ridge, Random Forest, and LSTM-GRU
on the 24h horizon (representative middle horizon).

Permutation importance = how much test RMSE INCREASES when we shuffle
a single feature column. High increase = feature was useful.

Outputs:
  - models/permutation_importance.csv  (all models, side by side)
  - models/permutation_importance.png  (top-15 bar chart)
  - MLflow: experiment "feature_importance"

Why this matters:
  Inspired by Sarkar et al. — gives a model-agnostic ranking of features
  to complement RF's SHAP plots. Useful for the final report's
  "feature analysis" section.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from sklearn.metrics import mean_squared_error

import mlflow
import dagshub

from src.training.data_loader import get_train_data, FORECAST_HORIZONS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = "aqi-predictor"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ANALYSIS_HORIZON = 24
N_REPEATS = 5
RNG = np.random.default_rng(42)

EXPERIMENT_NAME = "feature_importance"


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def load_models_from_registry():
    """Pull the registered Ridge, RF, CatBoost, XGBoost, and LSTM-GRU models
    for the 24h horizon."""
    client = mlflow.tracking.MlflowClient()
    models = {}

    # sklearn pipelines (Ridge, RF)
    for name in [f"ridge_{ANALYSIS_HORIZON}h", f"rf_{ANALYSIS_HORIZON}h"]:
        try:
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                logger.warning(f"  No registered versions for {name}")
                continue
            latest = max(versions, key=lambda mv: int(mv.version))
            model = mlflow.sklearn.load_model(f"models:/{name}/{latest.version}")
            models[name] = ("sklearn", model)
            logger.info(f"  Loaded {name} v{latest.version}")
        except Exception as e:
            logger.warning(f"  Could not load {name}: {e}")

    # CatBoost
    try:
        name = f"catboost_{ANALYSIS_HORIZON}h"
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            latest = max(versions, key=lambda mv: int(mv.version))
            model = mlflow.catboost.load_model(f"models:/{name}/{latest.version}")
            models[name] = ("catboost", model)
            logger.info(f"  Loaded {name} v{latest.version}")
    except Exception as e:
        logger.warning(f"  Could not load catboost_{ANALYSIS_HORIZON}h: {e}")

    # XGBoost
    try:
        name = f"xgboost_{ANALYSIS_HORIZON}h"
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            latest = max(versions, key=lambda mv: int(mv.version))
            model = mlflow.xgboost.load_model(f"models:/{name}/{latest.version}")
            models[name] = ("xgboost", model)
            logger.info(f"  Loaded {name} v{latest.version}")
    except Exception as e:
        logger.warning(f"  Could not load xgboost_{ANALYSIS_HORIZON}h: {e}")

    # LSTM-GRU is multi-output
    try:
        versions = client.search_model_versions("name='lstm_multihorizon'")
        if versions:
            latest = max(versions, key=lambda mv: int(mv.version))
            model = mlflow.tensorflow.load_model(f"models:/lstm_multihorizon/{latest.version}")
            models["lstm_gru"] = ("tensorflow", model)
            logger.info(f"  Loaded lstm_multihorizon v{latest.version}")
    except Exception as e:
        logger.warning(f"  Could not load lstm_multihorizon: {e}")

    return models


def permutation_importance_sklearn(model, X_test, y_test, aqi_now, feature_names, n_repeats=5):
    """For sklearn models predicting delta. Returns dict: {feature: mean_rmse_increase}."""
    # Baseline RMSE
    pred_d = model.predict(X_test)
    pred_abs = np.clip(aqi_now + pred_d, 0, 500)
    y_abs = aqi_now + y_test
    baseline_rmse = np.sqrt(mean_squared_error(y_abs, pred_abs))

    logger.info(f"    Baseline RMSE: {baseline_rmse:.3f}")

    importances = {}
    for col_idx, feat_name in enumerate(feature_names):
        deltas = []
        for rep in range(n_repeats):
            X_perm = X_test.copy()
            shuffled = RNG.permutation(X_perm[:, col_idx])
            X_perm[:, col_idx] = shuffled
            pred_d_perm = model.predict(X_perm)
            pred_abs_perm = np.clip(aqi_now + pred_d_perm, 0, 500)
            rmse_perm = np.sqrt(mean_squared_error(y_abs, pred_abs_perm))
            deltas.append(rmse_perm - baseline_rmse)
        importances[feat_name] = float(np.mean(deltas))
    return importances, baseline_rmse


def permutation_importance_tabular(model, X_test_df, y_test, aqi_now, feature_names,
                                    n_repeats=5, model_kind="catboost"):
    """For CatBoost/XGBoost which need DataFrame input with proper dtypes."""
    pred_d = model.predict(X_test_df)
    pred_abs = np.clip(aqi_now + pred_d, 0, 500)
    y_abs = aqi_now + y_test
    baseline_rmse = float(np.sqrt(mean_squared_error(y_abs, pred_abs)))

    logger.info(f"    Baseline RMSE: {baseline_rmse:.3f}")

    importances = {}
    for col_name in feature_names:
        deltas = []
        for rep in range(n_repeats):
            X_perm = X_test_df.copy()
            # Shuffle preserving dtype (important for CatBoost category cols)
            original = X_perm[col_name].values
            shuffled = RNG.permutation(original)
            X_perm[col_name] = shuffled
            # Reapply dtype if categorical (XGBoost)
            if model_kind == "xgboost" and col_name in ["hour", "day", "month",
                                                         "day_of_week", "is_weekend"]:
                X_perm[col_name] = X_perm[col_name].astype("category")
            pred_d_perm = model.predict(X_perm)
            pred_abs_perm = np.clip(aqi_now + pred_d_perm, 0, 500)
            rmse_perm = float(np.sqrt(mean_squared_error(y_abs, pred_abs_perm)))
            deltas.append(rmse_perm - baseline_rmse)
        importances[col_name] = float(np.mean(deltas))
    return importances, baseline_rmse


def permutation_importance_lstm(model, X_test_seq, y_test_d_full, aqi_now_seq,
                                 feature_names, horizon_idx, n_repeats=5):
    """For LSTM predicting all 6 deltas. We score the 24h horizon (index 3)."""
    pred_d = model.predict(X_test_seq, verbose=0)
    pred_abs = np.clip(aqi_now_seq + pred_d[:, horizon_idx], 0, 500)
    y_abs = aqi_now_seq + y_test_d_full[:, horizon_idx]
    baseline_rmse = float(np.sqrt(mean_squared_error(y_abs, pred_abs)))

    logger.info(f"    Baseline RMSE: {baseline_rmse:.3f}")

    importances = {}
    for col_idx, feat_name in enumerate(feature_names):
        deltas = []
        for rep in range(n_repeats):
            X_perm = X_test_seq.copy()
            # Shuffle this feature across ALL timesteps + all sequences
            for t in range(X_perm.shape[1]):
                X_perm[:, t, col_idx] = RNG.permutation(X_perm[:, t, col_idx])
            pred_d_perm = model.predict(X_perm, verbose=0)
            pred_abs_perm = np.clip(aqi_now_seq + pred_d_perm[:, horizon_idx], 0, 500)
            rmse_perm = float(np.sqrt(mean_squared_error(y_abs, pred_abs_perm)))
            deltas.append(rmse_perm - baseline_rmse)
        importances[feat_name] = float(np.mean(deltas))
    return importances, baseline_rmse


def build_test_sequences_for_lstm(X_test_arr, y_test_d_full, aqi_now_arr, seq_len=24):
    """Same window logic as train_lstm.py — needed because LSTM expects sequences."""
    n = len(X_test_arr)
    n_windows = n - seq_len + 1
    if n_windows <= 0:
        return None, None, None
    X_seq = np.stack([X_test_arr[i:i + seq_len] for i in range(n_windows)])
    y_seq = y_test_d_full[seq_len - 1: seq_len - 1 + n_windows]
    aqi_seq = aqi_now_arr[seq_len - 1: seq_len - 1 + n_windows]
    return X_seq.astype(np.float32), y_seq.astype(np.float32), aqi_seq.astype(np.float32)


def plot_top_features(df, n_top=15, save_path=None):
    """Side-by-side horizontal bar chart of top features per model."""
    models = [c for c in df.columns if c != "feature"]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    for ax, mname in zip(axes, models):
        d = df[["feature", mname]].sort_values(mname, ascending=False).head(n_top)
        ax.barh(d["feature"][::-1], d[mname][::-1], color="steelblue")
        ax.set_title(f"{mname}: ΔRMSE per feature", pad=10)
        ax.set_xlabel("RMSE increase when shuffled")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(f"Permutation Importance — 24h horizon (higher = more important)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"  Saved {save_path}")
    plt.close(fig)


def main():
    init_dagshub()
    logger.info("Loading data + registered models...")
    data = get_train_data()
    models = load_models_from_registry()
    if not models:
        logger.error("No models loaded. Train Ridge/RF/LSTM first.")
        return

    feature_names = data["feature_names"]

    # Use the 24h test set as our reference dataset
    X_test = data[f"X_test_{ANALYSIS_HORIZON}"].copy()
    y_test_d = data[f"y_delta_test_{ANALYSIS_HORIZON}"].copy()
    aqi_now_test = data[f"aqi_now_test_{ANALYSIS_HORIZON}"].copy()

    # Drop rows with NaN features
    mask = X_test.notna().all(axis=1)
    X_test = X_test[mask]
    y_test_d = y_test_d[mask]
    aqi_now_test = aqi_now_test[mask]

    X_test_arr = X_test.values
    y_test_arr = y_test_d.values
    aqi_now_arr = aqi_now_test.values

    # Build multi-horizon y_delta for LSTM
    full_test_df = data["full_df"]
    # Reindex the multi-horizon targets to match our filtered X_test
    test_idx = X_test.index
    y_test_d_full = np.column_stack([
        full_test_df.loc[test_idx, f"aqi_delta_{h}h"].values
        for h in FORECAST_HORIZONS
    ])

    all_importances = {"feature": feature_names}
    baselines = {}

    for model_key, (kind, model) in models.items():
        logger.info(f"\nComputing permutation importance for {model_key}...")
        if kind in ("sklearn", "catboost", "xgboost"):
            # All tree models predict on a 2D array (no sequence dim)
            # For CatBoost/XGBoost, need to cast categorical columns appropriately
            if kind == "catboost":
                # CatBoost wants int categorical cols; X_test_arr from .values
                # may have promoted them to float, so build a DataFrame
                X_for_pred = X_test.copy()
                for col in ["hour", "day", "month", "day_of_week", "is_weekend"]:
                    if col in X_for_pred.columns:
                        X_for_pred[col] = X_for_pred[col].astype(int)
                imp_dict, baseline = permutation_importance_tabular(
                    model, X_for_pred, y_test_arr, aqi_now_arr, feature_names,
                    n_repeats=N_REPEATS, model_kind="catboost",
                )
            elif kind == "xgboost":
                X_for_pred = X_test.copy()
                for col in ["hour", "day", "month", "day_of_week", "is_weekend"]:
                    if col in X_for_pred.columns:
                        X_for_pred[col] = X_for_pred[col].astype("category")
                imp_dict, baseline = permutation_importance_tabular(
                    model, X_for_pred, y_test_arr, aqi_now_arr, feature_names,
                    n_repeats=N_REPEATS, model_kind="xgboost",
                )
            else:
                # sklearn — original numpy array path
                imp_dict, baseline = permutation_importance_sklearn(
                    model, X_test_arr, y_test_arr, aqi_now_arr, feature_names,
                    n_repeats=N_REPEATS,
                )
            all_importances[model_key] = [imp_dict[f] for f in feature_names]
            baselines[model_key] = baseline
        elif kind == "tensorflow":
            X_seq, y_seq_full, aqi_seq = build_test_sequences_for_lstm(
                X_test_arr, y_test_d_full, aqi_now_arr, seq_len=24)
            if X_seq is None:
                logger.warning(f"  Not enough rows for LSTM sequences")
                continue
            horizon_idx = FORECAST_HORIZONS.index(ANALYSIS_HORIZON)
            imp_dict, baseline = permutation_importance_lstm(
                model, X_seq, y_seq_full, aqi_seq, feature_names,
                horizon_idx, n_repeats=N_REPEATS,
            )
            all_importances[model_key] = [imp_dict[f] for f in feature_names]
            baselines[model_key] = baseline

    df = pd.DataFrame(all_importances)
    csv_path = MODELS_DIR / "permutation_importance.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved {csv_path}")

    print("\n" + "=" * 72)
    print(f"PERMUTATION IMPORTANCE — {ANALYSIS_HORIZON}h horizon")
    print("=" * 72)
    print(f"\nBaseline RMSE per model:")
    for k, v in baselines.items():
        print(f"  {k:<20} {v:.3f}")

    print(f"\nTop 15 features per model (ΔRMSE when shuffled):\n")
    for mkey in [c for c in df.columns if c != "feature"]:
        top = df[["feature", mkey]].sort_values(mkey, ascending=False).head(15)
        print(f"--- {mkey} ---")
        for _, row in top.iterrows():
            bar = "█" * max(1, int(row[mkey] * 5))
            print(f"  {row['feature']:<25} {row[mkey]:+.3f}  {bar}")
        print()

    # Plot
    plot_path = MODELS_DIR / "permutation_importance.png"
    plot_top_features(df, n_top=15, save_path=plot_path)

    # Log to MLflow
    with mlflow.start_run(run_name=f"permutation_importance_{ANALYSIS_HORIZON}h"):
        mlflow.log_param("horizon_h", ANALYSIS_HORIZON)
        mlflow.log_param("n_repeats", N_REPEATS)
        mlflow.log_param("models_analyzed", list(models.keys()))
        for k, v in baselines.items():
            mlflow.log_metric(f"baseline_RMSE_{k}", v)
        mlflow.log_artifact(str(csv_path))
        if plot_path.exists():
            mlflow.log_artifact(str(plot_path))
        logger.info("✅ Logged feature importance to MLflow")

    print(f"\n✅ DagsHub: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
