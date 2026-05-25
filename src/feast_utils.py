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
    Falls back to reading from live Parquet only if online store unavailable.
    """
    store = get_store()
    try:
        result = store.get_online_features(
            features=FEATURE_REFS,
            entity_rows=[{"city": city}],
        ).to_dict()

        # Flatten: "view_name__feature_name" → "feature_name"
        # Also handle "view_name:feature_name" format
        flat = {}
        for key, vals in result.items():
            if key == "city":
                continue
            # Extract just the feature name from "view__feature" or "view:feature"
            if "__" in key:
                feat_name = key.split("__", 1)[-1]
            elif ":" in key:
                feat_name = key.split(":", 1)[-1]
            else:
                feat_name = key
            val = vals[0] if isinstance(vals, list) else vals
            flat[feat_name] = val

        # Check if we got real non-null values
        non_null = {k: v for k, v in flat.items() if v is not None}
        if len(non_null) < 5:
            logger.warning(
                f"Online store returned only {len(non_null)} non-null values "
                f"— falling back to live Parquet"
            )
            return _fallback_from_parquet(city)

        logger.info(
            f"✅ Online features for {city}: "
            f"AQI={flat.get('aqi')}, temp={flat.get('temperature')}, "
            f"n_features={len(non_null)}"
        )
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
    """Push features directly to Feast online store using write_to_online_store.

    Uses write_to_online_store() instead of push() — more reliable across
    Feast versions and explicitly targets the online store.
    """
    store = get_store()
    try:
        import pytz
        df = df.copy()

        # Feast requires timezone-aware timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(pytz.UTC)

        # Cast numeric columns to float32 (matches schema dtype)
        for col in df.columns:
            if col not in ["timestamp", "city"] and df[col].dtype != object:
                try:
                    df[col] = df[col].astype("float32")
                except (ValueError, TypeError):
                    pass

        # Write to each feature view's online store directly
        feature_views = store.list_feature_views()
        written = 0
        for fv in feature_views:
            # Get features that belong to this view
            fv_feature_names = [f.name for f in fv.features]
            available = [c for c in fv_feature_names if c in df.columns]
            if not available:
                continue

            # Build subset df for this feature view
            cols_needed = ["timestamp", "city"] + available
            subset = df[[c for c in cols_needed if c in df.columns]].copy()

            try:
                store.write_to_online_store(
                    feature_view_name=fv.name,
                    df=subset,
                )
                written += 1
                logger.debug(f"  Wrote to {fv.name}: {available[:3]}...")
            except Exception as e:
                logger.debug(f"  Skipped {fv.name}: {e}")

        logger.info(f"✅ Pushed to {written}/{len(feature_views)} feature views in online store")

    except Exception as e:
        logger.warning(f"Feast push failed: {e}")
