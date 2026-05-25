"""
AQI Predictor - Phase 3.5b: Daily LSTM-GRU (companion to hourly)
==================================================================
Aggregates the cleaned hourly data to DAILY averages, then trains
LSTM-GRU to predict 1, 2, 3 days ahead.

Why this exists:
- Sarkar et al. (Env. Pollut. 2022) used daily averages and got R²=0.84
  with LSTM-GRU on Delhi data.
- For the report: "We replicated the daily-aggregation approach from
  Sarkar et al. and compared it to our hourly multi-horizon pipeline."
- Not a replacement for the hourly model — a complementary deliverable.

This trains a SEPARATE LSTM-GRU on daily data and logs it as a distinct
model: "lstm_gru_daily" (so the hourly LSTM-GRU at "lstm_multihorizon"
is unaffected).

Output:
  - models/daily_lstm_metrics.csv
  - MLflow: experiment "lstm_gru_daily"
  - Registered model: lstm_gru_daily
"""

import os
import logging
from pathlib import Path
from typing import Tuple

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

from src.training.evaluate import (
    compute_metrics, compute_metrics_by_category,
    print_evaluation_report, metrics_summary_row,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = "aqi-predictor"
CLEANED_PATH = Path("data/interim/aqi_features_cleaned.parquet")
DAILY_PATH = Path("data/interim/aqi_features_daily.parquet")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Daily-scale hyperparameters
FORECAST_HORIZONS_DAYS = [1, 2, 3]  # days ahead
SEQUENCE_LENGTH = 14                # 2 weeks of daily history per sample
LSTM_UNITS = 32                     # smaller model — only 730 daily samples
GRU_UNITS = 16
DENSE_UNITS = 16
DROPOUT = 0.3
L2_LAMBDA = 1e-4
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15

EXPERIMENT_NAME = "lstm_gru_daily"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {mlflow.get_tracking_uri()} | exp: {EXPERIMENT_NAME}")


def aggregate_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Group hourly readings by date, compute per-day mean for each pollutant.
    Recomputes day-level lag/rolling features and 1/2/3-day-ahead targets.
    """
    df = df_hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()

    # Base pollutants to average
    pollutants = [c for c in ["aqi", "pm25", "pm10", "no2", "o3", "co", "so2"]
                  if c in df.columns]

    daily = df.groupby("date")[pollutants].mean().reset_index()
    daily = daily.rename(columns={"date": "timestamp"})
    daily["city"] = "Karachi"

    # Add day-of-week / month features
    dt = daily["timestamp"]
    daily["day_of_week"] = dt.dt.dayofweek.astype(int)
    daily["month"] = dt.dt.month.astype(int)
    daily["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)

    # Daily lag/rolling features for AQI
    daily["aqi_lag_1d"] = daily["aqi"].shift(1)
    daily["aqi_lag_3d"] = daily["aqi"].shift(3)
    daily["aqi_lag_7d"] = daily["aqi"].shift(7)
    daily["aqi_rolling_3d"] = daily["aqi"].rolling(3, min_periods=3).mean()
    daily["aqi_rolling_7d"] = daily["aqi"].rolling(7, min_periods=7).mean()
    daily["aqi_change_1d"] = daily["aqi"].diff(1)

    # Targets — both absolute and delta
    for h in FORECAST_HORIZONS_DAYS:
        daily[f"aqi_t_plus_{h}d"] = daily["aqi"].shift(-h)
        daily[f"aqi_delta_{h}d"] = daily[f"aqi_t_plus_{h}d"] - daily["aqi"]

    # Drop boundary NaN rows
    feature_lag_cols = [c for c in daily.columns
                       if "_lag_" in c or "_rolling_" in c or "_change_" in c]
    daily = daily.dropna(subset=feature_lag_cols).reset_index(drop=True)
    logger.info(f"Aggregated to {len(daily):,} daily rows from {len(df):,} hourly rows")
    return daily


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
    """Smaller LSTM-GRU for the daily dataset (fewer samples → smaller model)."""
    inputs = Input(shape=(SEQUENCE_LENGTH, n_features), name="features")
    x = LSTM(LSTM_UNITS, return_sequences=True,
             kernel_regularizer=l2(L2_LAMBDA), name="lstm")(inputs)
    x = Dropout(DROPOUT, name="dropout_lstm")(x)
    x = GRU(GRU_UNITS, return_sequences=False,
            kernel_regularizer=l2(L2_LAMBDA), name="gru")(x)
    x = Dropout(DROPOUT, name="dropout_gru")(x)
    x = Dense(DENSE_UNITS, activation="relu",
              kernel_regularizer=l2(L2_LAMBDA), name="dense_hidden")(x)
    x = Dropout(DROPOUT, name="dropout_dense")(x)
    outputs = Dense(n_outputs, activation="linear", name="deltas")(x)
    model = Model(inputs=inputs, outputs=outputs, name="lstm_gru_daily")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="mse", metrics=["mae"])
    return model


def main():
    init_dagshub()
    logger.info(f"Reading {CLEANED_PATH}...")
    df_hourly = pd.read_parquet(CLEANED_PATH)
    daily = aggregate_to_daily(df_hourly)

    # Save daily Parquet for reference / dashboard reuse
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PATH, index=False)
    logger.info(f"Saved {DAILY_PATH}")

    # Pick features
    EXCLUDE = {"timestamp", "city"}
    abs_targets = [f"aqi_t_plus_{h}d" for h in FORECAST_HORIZONS_DAYS]
    delta_targets = [f"aqi_delta_{h}d" for h in FORECAST_HORIZONS_DAYS]
    feature_cols = [c for c in daily.columns
                    if c not in EXCLUDE
                    and c not in abs_targets
                    and c not in delta_targets
                    and daily[c].dtype != "object"]

    logger.info(f"Using {len(feature_cols)} daily features")

    # Chronological split (80/10/10)
    n = len(daily)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    daily_train = daily.iloc[:train_end]
    daily_val = daily.iloc[train_end:val_end]
    daily_test = daily.iloc[val_end:]

    logger.info(f"Splits: train={len(daily_train)}, val={len(daily_val)}, test={len(daily_test)}")

    def make_arrays(d):
        X = d[feature_cols].values
        y_deltas = np.column_stack([d[f"aqi_delta_{h}d"].values for h in FORECAST_HORIZONS_DAYS])
        y_abs = np.column_stack([d[f"aqi_t_plus_{h}d"].values for h in FORECAST_HORIZONS_DAYS])
        aqi_now = d["aqi"].values
        # Drop rows with NaN delta targets
        mask = ~np.isnan(y_deltas).any(axis=1)
        return X[mask], y_deltas[mask], aqi_now[mask], y_abs[mask]

    X_train, y_train_d, aqi_train, y_train_a = make_arrays(daily_train)
    X_val, y_val_d, aqi_val, y_val_a = make_arrays(daily_val)
    X_test, y_test_d, aqi_test, y_test_a = make_arrays(daily_test)

    logger.info(f"After drop-NaN: train={X_train.shape[0]}, val={X_val.shape[0]}, "
                f"test={X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq, y_train_seq, aqi_train_seq, y_train_a_seq = build_sequences(
        X_train_scaled, y_train_d, aqi_train, y_train_a, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq, aqi_val_seq, y_val_a_seq = build_sequences(
        X_val_scaled, y_val_d, aqi_val, y_val_a, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq, aqi_test_seq, y_test_a_seq = build_sequences(
        X_test_scaled, y_test_d, aqi_test, y_test_a, SEQUENCE_LENGTH)

    logger.info(f"Daily sequences: train={X_train_seq.shape}, val={X_val_seq.shape}, "
                f"test={X_test_seq.shape}")

    if X_train_seq.shape[0] < 50:
        logger.error(f"Too few daily training sequences ({X_train_seq.shape[0]}). "
                     f"Need at least 2 years of hourly data.")
        return

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = build_model(X_train_scaled.shape[1], len(FORECAST_HORIZONS_DAYS))
    logger.info("\nDaily LSTM-GRU architecture:")
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                          min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )

    pred_train_d = model.predict(X_train_seq, batch_size=BATCH_SIZE, verbose=0)
    pred_val_d   = model.predict(X_val_seq,   batch_size=BATCH_SIZE, verbose=0)
    pred_test_d  = model.predict(X_test_seq,  batch_size=BATCH_SIZE, verbose=0)
    pred_train_a = np.clip(aqi_train_seq[:, None] + pred_train_d, 0, 500)
    pred_val_a   = np.clip(aqi_val_seq[:, None]   + pred_val_d,   0, 500)
    pred_test_a  = np.clip(aqi_test_seq[:, None]  + pred_test_d,  0, 500)

    summaries = []
    cat_breakdowns = {}
    print("\n" + "=" * 72)
    print("DAILY LSTM-GRU — PER-HORIZON METRICS")
    print("=" * 72)
    for i, h in enumerate(FORECAST_HORIZONS_DAYS):
        m_train = compute_metrics(y_train_a_seq[:, i], pred_train_a[:, i])
        m_val   = compute_metrics(y_val_a_seq[:, i],   pred_val_a[:, i])
        m_test  = compute_metrics(y_test_a_seq[:, i],  pred_test_a[:, i])
        cat = compute_metrics_by_category(y_test_a_seq[:, i], pred_test_a[:, i])
        cat_breakdowns[h] = cat
        print_evaluation_report(f"Daily LSTM-GRU", f"{h}d", m_train, m_val, m_test, cat)
        summaries.append({
            "model": "Daily-LSTM-GRU",
            "horizon": f"{h}d",
            "train_RMSE": m_train["RMSE"], "train_R2": m_train["R2"],
            "val_RMSE": m_val["RMSE"], "val_R2": m_val["R2"],
            "test_RMSE": m_test["RMSE"], "test_MAE": m_test["MAE"], "test_R2": m_test["R2"],
            "n_train": m_train["n_samples"], "n_val": m_val["n_samples"], "n_test": m_test["n_samples"],
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(MODELS_DIR / "daily_lstm_metrics.csv", index=False)

    with mlflow.start_run(run_name="lstm_gru_daily"):
        mlflow.log_param("model_type", "LSTM-GRU-daily")
        mlflow.log_param("granularity", "daily")
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("lstm_units", LSTM_UNITS)
        mlflow.log_param("gru_units", GRU_UNITS)
        mlflow.log_param("dense_units", DENSE_UNITS)
        mlflow.log_param("dropout", DROPOUT)
        mlflow.log_param("l2_lambda", L2_LAMBDA)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_epochs", EPOCHS)
        mlflow.log_param("actual_epochs", len(history.history["loss"]))
        mlflow.log_param("n_features", X_train_scaled.shape[1])
        mlflow.log_param("horizons_days", str(FORECAST_HORIZONS_DAYS))

        for row in summaries:
            h = row["horizon"]
            mlflow.log_metric(f"test_RMSE_{h}", row["test_RMSE"])
            mlflow.log_metric(f"test_MAE_{h}", row["test_MAE"])
            mlflow.log_metric(f"test_R2_{h}", row["test_R2"])
            mlflow.log_metric(f"val_RMSE_{h}", row["val_RMSE"])
            mlflow.log_metric(f"val_R2_{h}", row["val_R2"])
        mlflow.log_metric("test_RMSE_avg", float(summary_df["test_RMSE"].mean()))
        mlflow.log_metric("test_R2_avg",   float(summary_df["test_R2"].mean()))

        for ep, (loss, val_loss) in enumerate(zip(history.history["loss"],
                                                   history.history["val_loss"])):
            mlflow.log_metric("train_loss", loss, step=ep)
            mlflow.log_metric("val_loss", val_loss, step=ep)

        for h, cat in cat_breakdowns.items():
            p = MODELS_DIR / f"daily_lstm_{h}d_by_category.csv"
            cat.to_csv(p)
            mlflow.log_artifact(str(p))

        import joblib
        scaler_path = MODELS_DIR / "daily_lstm_scaler.pkl"
        joblib.dump({"scaler": scaler, "feature_cols": feature_cols,
                     "sequence_length": SEQUENCE_LENGTH,
                     "horizons_days": FORECAST_HORIZONS_DAYS}, scaler_path)
        mlflow.log_artifact(str(scaler_path))

        mlflow.tensorflow.log_model(
            model,
            artifact_path="lstm_gru_daily",
            registered_model_name="lstm_gru_daily",
        )
        logger.info("✅ Logged daily LSTM-GRU to MLflow")

    print("\n" + "=" * 72)
    print("DAILY LSTM-GRU — SUMMARY")
    print("=" * 72)
    print(summary_df[["model", "horizon", "val_RMSE", "val_R2",
                      "test_RMSE", "test_MAE", "test_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
