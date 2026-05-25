"""
AQI Predictor - Phase 2.5: Feature Imputation + Engineering
=============================================================
Reads raw backfill Parquet (AQ + weather), reindexes to hourly grid,
imputes gaps, then engineers all features used by training.

Feature groups (110 total with weather):
  Pollutants (8):       aqi, pm25, pm10, no2, o3, co, so2, nh3
  Weather (6):          temperature, humidity, precipitation,
                        wind_speed, pressure, cloud_cover
  Time (9):             hour, day, month, day_of_week, is_weekend,
                        hour_sin/cos, month_sin/cos
  AQI lags (7):         lag 1,3,6,12,24,48,72h
  AQI rolling (5):      rolling 3,6,24,48,72h mean
  AQI change (2):       diff 1h, 3h
  Pollutant lags (24):  {pm25..so2} lag_1h, lag_3h, rolling_3h, rolling_24h
  Weather lags (30):    {temp..cloud} lag_1h, lag_3h, lag_6h, lag_12h, lag_24h
  Weather rolling (18): {temp..cloud} rolling_3h, rolling_6h, rolling_24h
  Weather change (6):   {temp..cloud} change_1h
  Targets (12):         aqi_t_plus_{1,6,12,24,48,72}h + aqi_delta_{...}h
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH     = Path("data/raw/aqi_features_historical.parquet")
OUTPUT_PATH  = Path("data/interim/aqi_features_cleaned.parquet")

FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]

POLLUTANTS   = ["aqi","pm25","pm10","no2","o3","co","so2"]
WEATHER_COLS = ["temperature","humidity","precipitation","wind_speed","pressure","cloud_cover"]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded raw: {len(df):,} rows × {len(df.columns)} cols")
    return df


def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any missing hours with NaN rows so lags are always 1h apart."""
    full_range = pd.date_range(df["timestamp"].min(),
                               df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"
    df = df.reset_index()
    n_gaps = df["aqi"].isna().sum()
    if n_gaps > 0:
        logger.info(f"Reindexed: added {n_gaps} gap hours")
    df["city"] = df["city"].fillna("Karachi")
    return df


def impute_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short gaps (≤6h), then median for longer ones."""
    imputable = POLLUTANTS + WEATHER_COLS

    # Flag imputed rows before filling
    null_mask = df[imputable].isna().any(axis=1)
    df["is_imputed"] = null_mask.astype(int)

    # Flag long gaps (>6 consecutive hours)
    gap_len = null_mask.groupby((~null_mask).cumsum()).transform("sum")
    df["is_long_gap"] = (gap_len > 6).astype(int)

    # Forward fill up to 6h
    df[imputable] = df[imputable].ffill(limit=6)

    # Median fill remaining
    imp = SimpleImputer(strategy="median")
    df[imputable] = imp.fit_transform(df[imputable])
    logger.info(f"Imputed: {null_mask.sum()} rows "
                f"({null_mask.mean()*100:.1f}% of data)")
    return df


def clip_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physical lower bounds to gas concentrations.
    Negative gas concentrations are impossible — these are
    sensor/model artifacts (7 rows in 2 years of data).
    AQI and PM already have no negatives so we only clip gases.
    """
    # Gases cannot be negative — floor at 0
    for col in ["no2", "o3", "co", "so2"]:
        if col in df.columns:
            n_clipped = (df[col] < 0).sum()
            if n_clipped > 0:
                logger.info(f"  Clipping {n_clipped} negative {col} values to 0")
            df[col] = df[col].clip(lower=0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["timestamp"]
    df["hour"]        = dt.dt.hour.astype(int)
    df["day"]         = dt.dt.day.astype(int)
    df["month"]       = dt.dt.month.astype(int)
    df["day_of_week"] = dt.dt.dayofweek.astype(int)
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_aqi_lags(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)
    for w in [3, 6, 24, 48, 72]:
        df[f"aqi_rolling_{w}h"] = df["aqi"].rolling(w, min_periods=w).mean()
    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_3h"] = df["aqi"].diff(3)
    return df


def add_pollutant_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-only for pollutants — no rolling windows.
    Empirical test showed pollutant rolling features hurt at 48h/72h
    due to multicollinearity with AQI rolling features.
    Each lag captures a distinct past snapshot.
    """
    for p in ["pm25", "pm10", "no2", "o3", "co", "so2"]:
        df[f"{p}_lag_1h"]  = df[p].shift(1)
        df[f"{p}_lag_3h"]  = df[p].shift(3)
        df[f"{p}_lag_6h"]  = df[p].shift(6)
        df[f"{p}_lag_24h"] = df[p].shift(24)
    return df


def add_weather_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-only + change features for weather — no rolling windows.
    Lags: 1h (current trend), 6h (half-day), 12h (inversion cycle), 24h (daily context).
    Change features capture direction of movement, which matters more
    than absolute value for predicting future AQI patterns.
    """
    for col in WEATHER_COLS:
        if col not in df.columns:
            continue
        for lag in [1, 6, 12, 24]:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
        df[f"{col}_change_1h"] = df[col].diff(1)
        df[f"{col}_change_6h"] = df[col].diff(6)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    for h in FORECAST_HORIZONS:
        df[f"aqi_t_plus_{h}h"] = df["aqi"].shift(-h)
        df[f"aqi_delta_{h}h"]  = df[f"aqi_t_plus_{h}h"] - df["aqi"]
    return df


def drop_boundary_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the longest lag/rolling features are NaN.
    72h warmup needed for aqi_rolling_72h and aqi_lag_72h."""
    before = len(df)
    df = df.iloc[72:].reset_index(drop=True)
    logger.info(f"Dropped boundary rows: {before} → {len(df):,}")
    return df


def auto_drop_null_cols(df: pd.DataFrame, threshold: float = 0.999) -> pd.DataFrame:
    """Drop feature columns that are >99.9% null (e.g. nh3)."""
    EXCLUDE = {"timestamp","city","is_imputed","is_long_gap"}
    TARGET  = {c for c in df.columns if "aqi_t_plus" in c or "aqi_delta" in c}
    candidates = [c for c in df.columns if c not in EXCLUDE|TARGET]
    null_fracs = df[candidates].isna().mean()
    drop_cols  = null_fracs[null_fracs > threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Auto-dropped {len(drop_cols)} near-null cols: {drop_cols}")
    return df


def main():
    logger.info("Starting feature imputation + engineering")

    df = load_raw(RAW_PATH)
    df = reindex_hourly(df)
    df = impute_columns(df)
    df = clip_physical_bounds(df)
    df = add_time_features(df)
    df = add_aqi_lags(df)
    df = add_pollutant_lags(df)
    df = add_weather_lags(df)
    df = add_targets(df)
    df = drop_boundary_nans(df)
    df = auto_drop_null_cols(df)

    # Report final feature count
    EXCLUDE = {"timestamp","city","is_imputed","is_long_gap"}
    TARGET  = {c for c in df.columns if "aqi_t_plus" in c or "aqi_delta" in c}
    features = [c for c in df.columns if c not in EXCLUDE|TARGET and df[c].dtype!="object"]
    logger.info(f"\nFinal feature count: {len(features)}")
    logger.info(f"Total columns:       {len(df.columns)}")
    logger.info(f"Rows:                {len(df):,}")

    # Weather coverage report
    weather_in_features = [c for c in features if any(w in c for w in WEATHER_COLS)]
    logger.info(f"Weather features:    {len(weather_in_features)}")
    for w in WEATHER_COLS:
        w_feats = [c for c in features if c.startswith(w)]
        null_pct = df[w_feats].isna().mean().mean() * 100 if w_feats else 100
        logger.info(f"  {w:<15} → {len(w_feats)} features, {null_pct:.1f}% null")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    logger.info(f"\n✅ Saved {OUTPUT_PATH} ({size_mb:.2f} MB)")
    logger.info("Next: python -m src.training.train_ridge (etc.)")


if __name__ == "__main__":
    main()
