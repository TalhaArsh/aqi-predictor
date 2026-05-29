"""
AQI Predictor - Phase 3.1: Data Loader
========================================
Reads features for training. Supports two modes:

  1. Feast mode (USE_FEAST=true in .env):
     Reads from Feast offline store using point-in-time correct joins.
     Ensures no future data leakage, consistent feature schema.

  2. Parquet mode (default fallback):
     Reads directly from cleaned Parquet. Faster for development.
     Used automatically if Feast is not set up or USE_FEAST=false.

Uses DELTA targets (aqi_delta_{h}h) for training. At inference:
  forecast = current_aqi + predicted_delta
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLEANED_PATH = Path("data/interim/aqi_features_cleaned.parquet")

# Set USE_FEAST=true in .env to read from Feast offline store
USE_FEAST = os.getenv("USE_FEAST", "false").lower() == "true"

FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]

EXCLUDE_FROM_FEATURES = ["timestamp", "city", "is_imputed", "is_long_gap"]
ABSOLUTE_TARGET_COLS = [f"aqi_t_plus_{h}h" for h in FORECAST_HORIZONS]
DELTA_TARGET_COLS = [f"aqi_delta_{h}h" for h in FORECAST_HORIZONS]
ALL_TARGET_COLS = ABSOLUTE_TARGET_COLS + DELTA_TARGET_COLS


def load_data() -> pd.DataFrame:
    """Load feature data from Feast (if configured) or Parquet fallback."""
    if USE_FEAST:
        return _load_from_feast()
    return _load_from_parquet()


def _load_from_parquet() -> pd.DataFrame:
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"{CLEANED_PATH} not found. Run impute_features.py first.")
    df = pd.read_parquet(CLEANED_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Note: 90-day window is applied in impute_features.py (upstream)
    # No additional filtering needed here — cleaned Parquet already has
    # the correct date range.

    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {CLEANED_PATH}")
    fully_null = [c for c in df.columns if df[c].isna().all()]
    if fully_null:
        df = df.drop(columns=fully_null)
    return df


def _load_from_feast() -> pd.DataFrame:
    """Load training features from Feast offline store.
    Uses point-in-time correct joins — no future leakage.
    """
    logger.info("Loading features from Feast offline store...")
    try:
        from src.feast_utils import get_historical_features
        # Build entity dataframe from parquet timestamps
        base = pd.read_parquet(CLEANED_PATH)[["timestamp", "city",
                                               *ABSOLUTE_TARGET_COLS,
                                               *DELTA_TARGET_COLS]].copy()
        base["timestamp"] = pd.to_datetime(base["timestamp"])
        if base["timestamp"].dt.tz is None:
            import pytz
            base["timestamp"] = base["timestamp"].dt.tz_localize(pytz.UTC)
        entity_df = base[["timestamp", "city"]].copy()
        features = get_historical_features(entity_df)
        # Merge targets back in
        features = features.merge(
            base.drop(columns=["city"]), on="timestamp", how="left")
        features = features.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Feast: {len(features):,} rows × {len(features.columns)} cols")
        return features
    except Exception as e:
        logger.warning(f"Feast load failed: {e} — falling back to Parquet")
        return _load_from_parquet()


def chronological_split(df, train_frac=0.80, val_frac=0.10):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    logger.info("Chronological split:")
    logger.info(f"  Train: {len(train):,} ({train['timestamp'].min()} → {train['timestamp'].max()})")
    logger.info(f"  Val:   {len(val):,} ({val['timestamp'].min()} → {val['timestamp'].max()})")
    logger.info(f"  Test:  {len(test):,} ({test['timestamp'].min()} → {test['timestamp'].max()})")
    return train, val, test


def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and c not in ALL_TARGET_COLS
        and df[c].dtype != "object"
    ]


def split_features_targets(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].copy()
    y_delta = {h: df[f"aqi_delta_{h}h"].copy() for h in FORECAST_HORIZONS}
    y_abs = {h: df[f"aqi_t_plus_{h}h"].copy() for h in FORECAST_HORIZONS}
    aqi_now = df["aqi"].copy()
    return X, y_delta, y_abs, aqi_now


def drop_rows_missing_delta(X, y_delta, y_abs, aqi_now, horizon):
    mask = y_delta[horizon].notna()
    return (X[mask].copy(), y_delta[horizon][mask].copy(),
            y_abs[horizon][mask].copy(), aqi_now[mask].copy())


def get_train_data() -> dict:
    df = load_data()
    missing = [c for c in ALL_TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Targets missing: {missing}. Re-run impute_features.py.")

    train_df, val_df, test_df = chronological_split(df)
    feature_cols = pick_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} feature columns")

    out = {"feature_names": feature_cols, "full_df": df}
    for h in FORECAST_HORIZONS:
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            X, y_d, y_a, aqi_now = split_features_targets(split_df, feature_cols)
            X, y_d, y_a, aqi_now = drop_rows_missing_delta(X, y_d, y_a, aqi_now, h)
            out[f"X_{split_name}_{h}"] = X
            out[f"y_delta_{split_name}_{h}"] = y_d
            out[f"y_abs_{split_name}_{h}"] = y_a
            out[f"aqi_now_{split_name}_{h}"] = aqi_now
        logger.info(
            f"  {h}h horizon: train={len(out[f'X_train_{h}']):,}, "
            f"val={len(out[f'X_val_{h}']):,}, test={len(out[f'X_test_{h}']):,}"
        )

    out["train_timestamps"] = train_df["timestamp"].reset_index(drop=True)
    out["val_timestamps"] = val_df["timestamp"].reset_index(drop=True)
    out["test_timestamps"] = test_df["timestamp"].reset_index(drop=True)
    return out


def main():
    data = get_train_data()
    print("\n" + "=" * 72)
    print("DATA LOADER SUMMARY")
    print("=" * 72)
    print(f"\nFeatures ({len(data['feature_names'])}):")
    for i, c in enumerate(data['feature_names'], 1):
        print(f"  {i:>2}. {c}")
    print(f"\nSplit sizes per horizon:")
    print(f"  {'Horizon':<10} {'Train':>10} {'Val':>10} {'Test':>10}")
    for h in FORECAST_HORIZONS:
        print(f"  {h}h{'':<7} {len(data[f'X_train_{h}']):>10,} "
              f"{len(data[f'X_val_{h}']):>10,} {len(data[f'X_test_{h}']):>10,}")
    print(f"\nDelta target distribution (training labels):")
    for h in FORECAST_HORIZONS:
        y = data[f"y_delta_train_{h}"]
        print(f"  y_delta_train_{h}h: mean={y.mean():+.2f}, std={y.std():.1f}, "
              f"range=[{y.min():+.0f}, {y.max():+.0f}]")
    print("\n✅ Data loader ready.")


if __name__ == "__main__":
    main()
