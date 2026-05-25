"""
AQI Predictor - Phase 3.5: LSTM-GRU Hybrid (v4 — Sarkar et al. inspired)
=========================================================================
Hybrid architecture inspired by Sarkar et al. (Env. Pollut. 2022).

Architecture:
  Input: (24 timesteps, n_features)
  ├─ LSTM(64, return_sequences=True)         ← long-term dependencies
  ├─ Dropout(0.3)
  ├─ GRU(32, return_sequences=False)          ← compresses sequence + nonlinearity
  ├─ Dropout(0.3)
  ├─ Dense(32, relu) + L2(1e-4)
  ├─ Dropout(0.3)
  └─ Dense(6, linear)                         ← 6 delta outputs

Why hybrid:
  Sarkar et al. found LSTM-GRU stacks beat standalone LSTM (R²: 0.78 → 0.84).
  LSTM captures long-range; GRU adds a faster nonlinear transformation
  with fewer parameters than another LSTM layer.

Kept from v3:
  - Walk-forward cross-validation (5 folds) for honest seasonal averaging
  - Heavy regularization (dropout 0.3, L2 weight decay)
  - Delta-target framing
  - Same holdout test for apples-to-apples comparison

Outputs:
  - models/lstm_metrics.csv          (per-horizon final test metrics)
  - models/lstm_cv_metrics.csv       (per-horizon CV mean ± std)
  - MLflow: experiment "lstm_gru_hybrid", run "lstm_gru_hybrid_v4"
  - Registered model: lstm_multihorizon (replaces v3)
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.tensorflow
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

# v4 LSTM-GRU hybrid hyperparameters
SEQUENCE_LENGTH = 24
LSTM_UNITS = 64           # first layer: captures temporal patterns
GRU_UNITS = 32            # second layer: nonlinear compression
DENSE_UNITS = 32
DROPOUT = 0.3
L2_LAMBDA = 1e-4
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10

N_CV_FOLDS = 5
TEST_HOLDOUT_FRAC = 0.10

EXPERIMENT_NAME = "lstm_gru_hybrid"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def build_sequences(X, y_deltas, aqi_now, y_abs, seq_len):
    n = len(X)
    n_windows = n - seq_len + 1
    if n_windows <= 0:
        return (np.empty((0, seq_len, X.shape[1])),
                np.empty((0, y_deltas.shape[1])),
                np.empty(0), np.empty((0, y_deltas.shape[1])))
    X_seq = np.stack([X[i:i + seq_len] for i in range(n_windows)])
    y_seq = y_deltas[seq_len - 1: seq_len - 1 + n_windows]
    aqi_seq = aqi_now[seq_len - 1: seq_len - 1 + n_windows]
    y_abs_seq = y_abs[seq_len - 1: seq_len - 1 + n_windows]
    return X_seq.astype(np.float32), y_seq.astype(np.float32), \
           aqi_seq.astype(np.float32), y_abs_seq.astype(np.float32)


def build_model(n_features, n_outputs):
    """LSTM → GRU hybrid (Sarkar et al. inspired)."""
    inputs = Input(shape=(SEQUENCE_LENGTH, n_features), name="features")

    # LSTM layer captures long-term temporal dependencies, returns sequence
    x = LSTM(
        LSTM_UNITS,
        return_sequences=True,
        kernel_regularizer=l2(L2_LAMBDA),
        name="lstm",
    )(inputs)
    x = Dropout(DROPOUT, name="dropout_lstm")(x)

    # GRU compresses the sequence into a single context vector
    x = GRU(
        GRU_UNITS,
        return_sequences=False,
        kernel_regularizer=l2(L2_LAMBDA),
        name="gru",
    )(x)
    x = Dropout(DROPOUT, name="dropout_gru")(x)

    # Dense bottleneck
    x = Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=l2(L2_LAMBDA),
        name="dense_hidden",
    )(x)
    x = Dropout(DROPOUT, name="dropout_dense")(x)

    outputs = Dense(n_outputs, activation="linear", name="deltas")(x)

    model = Model(inputs=inputs, outputs=outputs, name="lstm_gru_hybrid")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def prepare_full_data(data):
    feature_cols = data["feature_names"]
    full_df = data["full_df"]

    X_df = full_df[feature_cols].copy()
    y_deltas = np.column_stack([full_df[f"aqi_delta_{h}h"].values for h in FORECAST_HORIZONS])
    y_abs = np.column_stack([full_df[f"aqi_t_plus_{h}h"].values for h in FORECAST_HORIZONS])
    aqi_now = full_df["aqi"].values

    mask = ~np.isnan(y_deltas).any(axis=1)
    X_df = X_df[mask].reset_index(drop=True)
    y_deltas = y_deltas[mask]
    y_abs = y_abs[mask]
    aqi_now = aqi_now[mask]
    logger.info(f"Full usable data: {len(X_df):,} rows")
    return X_df, y_deltas, aqi_now, y_abs, feature_cols


def walkforward_cv_indices(n_total, n_folds, holdout_frac):
    n_holdout = int(n_total * holdout_frac)
    n_cv = n_total - n_holdout
    fold_size = n_cv // (n_folds + 1)
    initial_train = fold_size
    folds = []
    for k in range(n_folds):
        train_end = initial_train + k * fold_size
        val_start = train_end
        val_end = val_start + fold_size
        if val_end > n_cv:
            val_end = n_cv
        if val_start >= val_end:
            break
        folds.append((slice(0, train_end), slice(val_start, val_end)))
    return folds, n_cv, n_holdout


def train_one_fold(X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_features, fold_idx):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + fold_idx)
    model = build_model(n_features, len(FORECAST_HORIZONS))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-5, verbose=0),
    ]
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )
    return model, min(history.history["val_loss"]), len(history.history["loss"])


def main():
    init_dagshub()
    logger.info("Loading data...")
    data = get_train_data()

    X_df, y_deltas, aqi_now, y_abs, feature_cols = prepare_full_data(data)
    n_total = len(X_df)

    folds, n_cv, n_holdout = walkforward_cv_indices(n_total, N_CV_FOLDS, TEST_HOLDOUT_FRAC)
    logger.info(f"\nWalk-forward CV setup:")
    logger.info(f"  Total usable: {n_total:,}")
    logger.info(f"  CV data:      {n_cv:,}")
    logger.info(f"  Test holdout: {n_holdout:,}")
    for k, (tr_s, va_s) in enumerate(folds, 1):
        logger.info(f"    Fold {k}: train [0:{tr_s.stop}], val [{va_s.start}:{va_s.stop}]")

    # CV phase
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION (5 folds, LSTM-GRU hybrid)")
    logger.info("=" * 70)

    cv_metrics = {h: {"val_RMSE": [], "val_R2": []} for h in FORECAST_HORIZONS}
    fold_logs = []

    for k, (train_slc, val_slc) in enumerate(folds, 1):
        logger.info(f"\n--- Fold {k}/{len(folds)} ---")

        X_train_arr = X_df.iloc[train_slc].values
        y_train_d = y_deltas[train_slc]
        aqi_train = aqi_now[train_slc]
        y_train_a = y_abs[train_slc]

        X_val_arr = X_df.iloc[val_slc].values
        y_val_d = y_deltas[val_slc]
        aqi_val = aqi_now[val_slc]
        y_val_a = y_abs[val_slc]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_arr)
        X_val_scaled = scaler.transform(X_val_arr)

        X_train_seq, y_train_seq, _, _ = build_sequences(
            X_train_scaled, y_train_d, aqi_train, y_train_a, SEQUENCE_LENGTH)
        X_val_seq, y_val_seq, aqi_val_seq, y_val_a_seq = build_sequences(
            X_val_scaled, y_val_d, aqi_val, y_val_a, SEQUENCE_LENGTH)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            logger.warning(f"  Fold {k}: empty sequences, skipping")
            continue

        logger.info(f"  Sequences: train={X_train_seq.shape}, val={X_val_seq.shape}")

        model, best_val_loss, ep = train_one_fold(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            X_train_arr.shape[1], k,
        )
        logger.info(f"  Trained {ep} epochs, best val_loss={best_val_loss:.2f}")

        pred_val_d = model.predict(X_val_seq, batch_size=BATCH_SIZE, verbose=0)
        pred_val_a = np.clip(aqi_val_seq[:, None] + pred_val_d, 0, 500)

        fold_log = {"fold": k}
        for i, h in enumerate(FORECAST_HORIZONS):
            m = compute_metrics(y_val_a_seq[:, i], pred_val_a[:, i])
            cv_metrics[h]["val_RMSE"].append(m["RMSE"])
            cv_metrics[h]["val_R2"].append(m["R2"])
            fold_log[f"val_RMSE_{h}h"] = m["RMSE"]
            fold_log[f"val_R2_{h}h"] = m["R2"]
        fold_logs.append(fold_log)
        logger.info(f"  Val RMSE: " + ", ".join(
            f"{h}h={fold_log[f'val_RMSE_{h}h']:.1f}" for h in FORECAST_HORIZONS))

    print("\n" + "=" * 72)
    print("CV SUMMARY — LSTM-GRU hybrid (mean ± std across folds)")
    print("=" * 72)
    print(f"\n{'Horizon':<10} {'Val RMSE':<22} {'Val R²':<22}")
    print("-" * 56)
    cv_summary = []
    for h in FORECAST_HORIZONS:
        rmses = cv_metrics[h]["val_RMSE"]
        r2s = cv_metrics[h]["val_R2"]
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        r2_mean, r2_std = np.mean(r2s), np.std(r2s)
        cv_summary.append({
            "horizon_h": h,
            "cv_val_RMSE_mean": rmse_mean, "cv_val_RMSE_std": rmse_std,
            "cv_val_R2_mean": r2_mean, "cv_val_R2_std": r2_std,
        })
        print(f"  {h}h{'':<7} {rmse_mean:>6.2f} ± {rmse_std:<5.2f}{'':<7} "
              f"{r2_mean:>+.3f} ± {r2_std:.3f}")

    # FINAL TRAINING
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TRAINING — full CV window, holdout test eval")
    logger.info("=" * 70)

    train_end = n_cv
    X_train_full = X_df.iloc[:train_end].values
    y_train_d_full = y_deltas[:train_end]
    aqi_train_full = aqi_now[:train_end]
    y_train_a_full = y_abs[:train_end]

    X_test = X_df.iloc[train_end:].values
    y_test_d = y_deltas[train_end:]
    aqi_test = aqi_now[train_end:]
    y_test_a = y_abs[train_end:]

    val_split = int(train_end * 0.9)
    X_train_arr = X_train_full[:val_split]
    y_train_d = y_train_d_full[:val_split]
    aqi_train = aqi_train_full[:val_split]
    y_train_a = y_train_a_full[:val_split]
    X_val_arr = X_train_full[val_split:]
    y_val_d = y_train_d_full[val_split:]
    aqi_val = aqi_train_full[val_split:]
    y_val_a = y_train_a_full[val_split:]

    logger.info(f"  Train: {len(X_train_arr):,}, Val: {len(X_val_arr):,}, Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_val_scaled = scaler.transform(X_val_arr)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq, y_train_seq, aqi_train_seq, y_train_a_seq = build_sequences(
        X_train_scaled, y_train_d, aqi_train, y_train_a, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq, aqi_val_seq, y_val_a_seq = build_sequences(
        X_val_scaled, y_val_d, aqi_val, y_val_a, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq, aqi_test_seq, y_test_a_seq = build_sequences(
        X_test_scaled, y_test_d, aqi_test, y_test_a, SEQUENCE_LENGTH)
    logger.info(f"  Sequences: train={X_train_seq.shape}, val={X_val_seq.shape}, "
                f"test={X_test_seq.shape}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    final_model = build_model(X_train_scaled.shape[1], len(FORECAST_HORIZONS))
    logger.info("\nLSTM-GRU hybrid architecture:")
    final_model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-5, verbose=1),
    ]
    history = final_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )

    pred_train_d = final_model.predict(X_train_seq, batch_size=BATCH_SIZE, verbose=0)
    pred_val_d   = final_model.predict(X_val_seq,   batch_size=BATCH_SIZE, verbose=0)
    pred_test_d  = final_model.predict(X_test_seq,  batch_size=BATCH_SIZE, verbose=0)
    pred_train_a = np.clip(aqi_train_seq[:, None] + pred_train_d, 0, 500)
    pred_val_a   = np.clip(aqi_val_seq[:, None]   + pred_val_d,   0, 500)
    pred_test_a  = np.clip(aqi_test_seq[:, None]  + pred_test_d,  0, 500)

    final_summaries = []
    cat_breakdowns = {}
    print("\n" + "=" * 72)
    print("LSTM-GRU HYBRID — FINAL TEST METRICS (absolute scale, holdout)")
    print("=" * 72)
    for i, h in enumerate(FORECAST_HORIZONS):
        m_train = compute_metrics(y_train_a_seq[:, i], pred_train_a[:, i])
        m_val   = compute_metrics(y_val_a_seq[:, i],   pred_val_a[:, i])
        m_test  = compute_metrics(y_test_a_seq[:, i],  pred_test_a[:, i])
        cat = compute_metrics_by_category(y_test_a_seq[:, i], pred_test_a[:, i])
        cat_breakdowns[h] = cat
        print_evaluation_report(f"LSTM-GRU", h, m_train, m_val, m_test, cat)
        final_summaries.append(metrics_summary_row("LSTM-GRU", h, m_train, m_val, m_test))

    summary_df = pd.DataFrame(final_summaries)
    summary_df.to_csv(MODELS_DIR / "lstm_metrics.csv", index=False)
    pd.DataFrame(cv_summary).to_csv(MODELS_DIR / "lstm_cv_metrics.csv", index=False)
    pd.DataFrame(fold_logs).to_csv(MODELS_DIR / "lstm_cv_fold_details.csv", index=False)

    with mlflow.start_run(run_name="lstm_gru_hybrid_v4"):
        mlflow.log_param("model_type", "LSTM-GRU-hybrid")
        mlflow.log_param("architecture", "LSTM→GRU→Dense")
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("lstm_units", LSTM_UNITS)
        mlflow.log_param("gru_units", GRU_UNITS)
        mlflow.log_param("dense_units", DENSE_UNITS)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("l2_lambda", L2_LAMBDA)
        mlflow.log_param("n_cv_folds", N_CV_FOLDS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_epochs", EPOCHS)
        mlflow.log_param("actual_epochs", len(history.history["loss"]))
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("n_features", X_train_scaled.shape[1])
        mlflow.log_param("horizons", str(FORECAST_HORIZONS))
        mlflow.log_param("target_type", "delta_multihorizon")

        for row in final_summaries:
            h = row["horizon_h"]
            mlflow.log_metric(f"test_RMSE_{h}h", row["test_RMSE"])
            mlflow.log_metric(f"test_MAE_{h}h", row["test_MAE"])
            mlflow.log_metric(f"test_R2_{h}h", row["test_R2"])
            mlflow.log_metric(f"val_RMSE_{h}h", row["val_RMSE"])
            mlflow.log_metric(f"val_R2_{h}h", row["val_R2"])
        for row in cv_summary:
            h = row["horizon_h"]
            mlflow.log_metric(f"cv_val_RMSE_mean_{h}h", row["cv_val_RMSE_mean"])
            mlflow.log_metric(f"cv_val_RMSE_std_{h}h", row["cv_val_RMSE_std"])
            mlflow.log_metric(f"cv_val_R2_mean_{h}h", row["cv_val_R2_mean"])
        mlflow.log_metric("test_RMSE_avg", float(summary_df["test_RMSE"].mean()))
        mlflow.log_metric("test_R2_avg",   float(summary_df["test_R2"].mean()))

        for ep, (loss, val_loss) in enumerate(zip(history.history["loss"],
                                                   history.history["val_loss"])):
            mlflow.log_metric("train_loss", loss, step=ep)
            mlflow.log_metric("val_loss", val_loss, step=ep)

        for h, cat in cat_breakdowns.items():
            p = MODELS_DIR / f"lstm_{h}h_by_category.csv"
            cat.to_csv(p)
            mlflow.log_artifact(str(p))
        mlflow.log_artifact(str(MODELS_DIR / "lstm_cv_metrics.csv"))
        mlflow.log_artifact(str(MODELS_DIR / "lstm_cv_fold_details.csv"))

        import joblib
        scaler_path = MODELS_DIR / "lstm_scaler.pkl"
        joblib.dump({"scaler": scaler, "feature_cols": feature_cols,
                     "sequence_length": SEQUENCE_LENGTH,
                     "horizons": FORECAST_HORIZONS}, scaler_path)
        mlflow.log_artifact(str(scaler_path))

        mlflow.tensorflow.log_model(
            final_model,
            artifact_path="lstm_multihorizon",
            registered_model_name="lstm_multihorizon",
        )
        logger.info("✅ Logged LSTM-GRU hybrid + CV + scaler to MLflow")

    print("\n" + "=" * 72)
    print("LSTM-GRU HYBRID v4 — SUMMARY")
    print("=" * 72)
    print(summary_df[["model", "horizon_h", "val_RMSE", "val_R2",
                      "test_RMSE", "test_MAE", "test_R2"]].to_string(index=False))
    print(f"\n✅ DagsHub: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments")


if __name__ == "__main__":
    main()
