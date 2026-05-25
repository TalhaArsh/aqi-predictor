"""
AQI Predictor — Feast Store Utilities
=======================================
Shared helpers for reading/writing features through Feast.
Used by feature_pipeline.py (writes), data_loader.py (reads historical),
and dashboard (reads online).

The key concept: Feast gives you point-in-time correct joins.
When training, you ask "what features were available at t=2024-06-01 12:00?"
and Feast returns the values as they existed at that exact timestamp —
not future values that would leak information.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Feast repo path — relative to project root
FEAST_REPO_PATH = Path("feature_store")

ALL_FEATURE_VIEWS = [
    "aqi_features",
    "weather_features",
    "time_features",
    "aqi_lag_features",
    "pollutant_lag_features",
    "weather_lag_features",
]

# All 96 feature names — pulled from all feature views
FEATURE_REFS = [
    # aqi_features
    "aqi_features:aqi", "aqi_features:pm25", "aqi_features:pm10",
    "aqi_features:no2", "aqi_features:o3", "aqi_features:co", "aqi_features:so2",
    # weather_features
    "weather_features:temperature", "weather_features:humidity",
    "weather_features:precipitation", "weather_features:wind_speed",
    "weather_features:pressure", "weather_features:cloud_cover",
    # time_features
    "time_features:hour", "time_features:day", "time_features:month",
    "time_features:day_of_week", "time_features:is_weekend",
    "time_features:hour_sin", "time_features:hour_cos",
    "time_features:month_sin", "time_features:month_cos",
    # aqi_lag_features
    "aqi_lag_features:aqi_lag_1h", "aqi_lag_features:aqi_lag_3h",
    "aqi_lag_features:aqi_lag_6h", "aqi_lag_features:aqi_lag_12h",
    "aqi_lag_features:aqi_lag_24h", "aqi_lag_features:aqi_lag_48h",
    "aqi_lag_features:aqi_lag_72h",
    "aqi_lag_features:aqi_rolling_3h", "aqi_lag_features:aqi_rolling_6h",
    "aqi_lag_features:aqi_rolling_24h", "aqi_lag_features:aqi_rolling_48h",
    "aqi_lag_features:aqi_rolling_72h",
    "aqi_lag_features:aqi_change_1h", "aqi_lag_features:aqi_change_3h",
    # pollutant_lag_features
    *[f"pollutant_lag_features:{p}_lag_{h}h"
      for p in ["pm25","pm10","no2","o3","co","so2"]
      for h in [1, 3, 6, 24]],
    # weather_lag_features
    *[f"weather_lag_features:{col}_lag_{h}h"
      for col in ["temperature","humidity","precipitation",
                  "wind_speed","pressure","cloud_cover"]
      for h in [1, 6, 12, 24]],
    *[f"weather_lag_features:{col}_change_{h}h"
      for col in ["temperature","humidity","precipitation",
                  "wind_speed","pressure","cloud_cover"]
      for h in [1, 6]],
]


def get_store():
    """Return initialized Feast FeatureStore.
    
    Builds Redis connection string from environment variables so
    credentials are never hardcoded. Works locally (.env) and in
    CI/CD (GitHub Actions secrets) and on Streamlit Cloud (secrets).
    """
    import os
    try:
        from feast import FeatureStore
        # Inject Redis connection string from env vars
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_pass = os.getenv("REDIS_PASSWORD", "")
        if redis_pass:
            conn_str = f"redis://:{redis_pass}@{redis_host}:{redis_port}"
        else:
            conn_str = f"redis://{redis_host}:{redis_port}"
        os.environ["REDIS_CONNECTION_STRING"] = conn_str
        return FeatureStore(repo_path=str(FEAST_REPO_PATH))
    except ImportError:
        raise ImportError(
            "Feast not installed. Run: pip install 'feast[redis]>=0.40.0'"
        )


def materialize_offline_to_online(start_date=None, end_date=None):
    """Push recent offline features to the online store.

    Called by feature_pipeline.py after writing new data.
    Makes latest features available for millisecond online serving.
    """
    from datetime import datetime, timezone, timedelta
    store = get_store()
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(hours=4)
    logger.info(f"Materializing offline → online: {start_date} → {end_date}")
    store.materialize(start_date=start_date, end_date=end_date)
    logger.info("✅ Online store updated")


def get_historical_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve features for training using point-in-time correct joins.

    entity_df must have columns: [timestamp, city]
    Returns: entity_df joined with all feature values at each timestamp

    This is the correct way to build training datasets — Feast ensures
    you only see feature values that existed before each timestamp,
    preventing future data leakage.
    """
    store = get_store()
    logger.info(f"Fetching historical features for {len(entity_df):,} entity rows")
    job = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    )
    df = job.to_df()
    logger.info(f"Retrieved: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_online_features(city: str = "Karachi") -> Optional[dict]:
    """Retrieve latest features from online store for real-time inference.

    Returns dict of {feature_name: value} for the given city.
    Used by the dashboard to get current features before calling the model.
    Falls back to reading from Parquet if online store is empty.
    """
    store = get_store()
    try:
        result = store.get_online_features(
            features=FEATURE_REFS,
            entity_rows=[{"city": city}],
        ).to_dict()

        # Check if we got real values (not all None)
        sample_val = result.get("aqi_features__aqi", [None])[0]
        if sample_val is None:
            logger.warning("Online store empty — falling back to Parquet")
            return _fallback_from_parquet(city)

        # Flatten: "view_name__feature_name" → "feature_name"
        flat = {}
        for key, vals in result.items():
            feat_name = key.split("__")[-1] if "__" in key else key
            flat[feat_name] = vals[0] if isinstance(vals, list) else vals

        logger.info(f"Online features retrieved for {city}: AQI={flat.get('aqi')}")
        return flat

    except Exception as e:
        logger.warning(f"Online store read failed: {e} — falling back to Parquet")
        return _fallback_from_parquet(city)


def _fallback_from_parquet(city: str) -> Optional[dict]:
    """Read latest row from live Parquet as fallback."""
    live_path = Path("data/raw/aqi_features_live.parquet")
    if not live_path.exists():
        logger.error("No live Parquet found either")
        return None
    df = pd.read_parquet(live_path)
    df = df[df["city"] == city].sort_values("timestamp").tail(1)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def push_to_online_store(df: pd.DataFrame):
    """Push a DataFrame of features directly to the online store.

    Called by feature_pipeline.py after each hourly fetch.
    df must have [timestamp, city] columns plus all feature columns.
    """
    store = get_store()
    try:
        store.push("aqi_push_source", df, to=store.PushMode.ONLINE_AND_OFFLINE)
        logger.info(f"Pushed {len(df)} row(s) to Feast online + offline store")
    except Exception as e:
        logger.warning(f"Feast push failed: {e}")
