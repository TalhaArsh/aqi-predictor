"""
AQI Predictor - Phase 1: Live Feature Pipeline
================================================
Fetches current AQ + weather from Open-Meteo (same source as training).
No API keys needed. Eliminates training-serving skew.

At inference time we fetch:
  - Current AQ (us_aqi, pm25, pm10, no2, o3, co, so2)
  - Current weather (temperature, humidity, precipitation,
                     wind_speed, pressure, cloud_cover)
  - Last 72h history from live Parquet for lag features
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CITY_NAME = os.getenv("CITY_NAME", "Karachi")
CITY_LAT  = float(os.getenv("CITY_LAT",  "24.8607"))
CITY_LON  = float(os.getenv("CITY_LON",  "67.0011"))

OUTPUT_PATH = Path("data/raw/aqi_features_live.parquet")

AQ_API  = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_API  = "https://api.open-meteo.com/v1/forecast"   # free, no key

AQ_VARS = ["us_aqi","pm2_5","pm10","nitrogen_dioxide",
           "ozone","carbon_monoxide","sulphur_dioxide","ammonia"]
WX_VARS = ["temperature_2m","relative_humidity_2m","precipitation",
           "wind_speed_10m","surface_pressure","cloud_cover"]

WEATHER_COLS = ["temperature","humidity","precipitation",
                "wind_speed","pressure","cloud_cover"]
POLLUTANTS   = ["pm25","pm10","no2","o3","co","so2"]


def fetch_current_aq() -> Optional[pd.DataFrame]:
    params = {"latitude": CITY_LAT, "longitude": CITY_LON,
              "hourly": ",".join(AQ_VARS),
              "forecast_days": 1, "timezone": "GMT"}
    try:
        r = requests.get(AQ_API, params=params, timeout=20)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(h["time"]),
            "aqi":  h.get("us_aqi"),
            "pm25": h.get("pm2_5"),
            "pm10": h.get("pm10"),
            "no2":  h.get("nitrogen_dioxide"),
            "o3":   h.get("ozone"),
            "co":   h.get("carbon_monoxide"),
            "so2":  h.get("sulphur_dioxide"),
            "nh3":  h.get("ammonia"),
        })
        for col in ["aqi","pm25","pm10","no2","o3","co","so2","nh3"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Most recent non-null AQI row
        valid = df.dropna(subset=["aqi"])
        row = valid.iloc[[-1]] if not valid.empty else df.iloc[[-1]]
        logger.info(f"AQ: timestamp={row['timestamp'].iloc[0]}, "
                    f"AQI={row['aqi'].iloc[0]}, PM2.5={row['pm25'].iloc[0]}")
        return row.reset_index(drop=True)
    except Exception as e:
        logger.error(f"AQ fetch failed: {e}")
        return None


def fetch_current_weather() -> Optional[pd.DataFrame]:
    """Open-Meteo forecast API — free, no key, returns current + forecast hours."""
    params = {"latitude": CITY_LAT, "longitude": CITY_LON,
              "hourly": ",".join(WX_VARS),
              "forecast_days": 1, "timezone": "UTC"}
    try:
        r = requests.get(WX_API, params=params, timeout=20)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "timestamp":    pd.to_datetime(h["time"]),
            "temperature":  h.get("temperature_2m"),
            "humidity":     h.get("relative_humidity_2m"),
            "precipitation":h.get("precipitation"),
            "wind_speed":   h.get("wind_speed_10m"),
            "pressure":     h.get("surface_pressure"),
            "cloud_cover":  h.get("cloud_cover"),
        })
        for col in WEATHER_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Match most recent available hour
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0,
                                                      microsecond=0, tzinfo=None)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        row = df[df["timestamp"] <= pd.Timestamp(now_utc)].iloc[[-1]]
        logger.info(f"WX: temp={row['temperature'].iloc[0]}°C, "
                    f"wind={row['wind_speed'].iloc[0]} km/h, "
                    f"precip={row['precipitation'].iloc[0]} mm")
        return row.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return None


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


def add_lag_features(df_new: pd.DataFrame,
                     df_history: pd.DataFrame) -> pd.DataFrame:
    """Compute lag/rolling features using last 72h of history.
    Matches impute_features.py feature definitions exactly."""
    if df_history.empty:
        logger.warning("No history — lag features will be NaN")
        return df_new

    # Ensure consistent timezone (strip tz from both if mixed)
    for _df in [df_history, df_new]:
        if "timestamp" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["timestamp"])
            if hasattr(_df["timestamp"].dt, "tz") and _df["timestamp"].dt.tz is not None:
                _df["timestamp"] = _df["timestamp"].dt.tz_localize(None)

    combined = pd.concat([df_history, df_new], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # AQI lags
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        combined[f"aqi_lag_{lag}h"] = combined["aqi"].shift(lag)
    for w in [3, 6, 24, 48, 72]:
        combined[f"aqi_rolling_{w}h"] = combined["aqi"].rolling(w, min_periods=1).mean()
    combined["aqi_change_1h"] = combined["aqi"].diff(1)
    combined["aqi_change_3h"] = combined["aqi"].diff(3)

    # Pollutant lags
    for p in POLLUTANTS:
        if p in combined.columns:
            combined[f"{p}_lag_1h"]      = combined[p].shift(1)
            combined[f"{p}_lag_3h"]      = combined[p].shift(3)
            combined[f"{p}_rolling_3h"]  = combined[p].rolling(3,  min_periods=1).mean()
            combined[f"{p}_rolling_24h"] = combined[p].rolling(24, min_periods=1).mean()

    # Weather lags
    for col in WEATHER_COLS:
        if col in combined.columns:
            for lag in [1, 3, 6, 12, 24]:
                combined[f"{col}_lag_{lag}h"] = combined[col].shift(lag)
            for w in [3, 6, 24]:
                combined[f"{col}_rolling_{w}h"] = combined[col].rolling(w, min_periods=1).mean()
            combined[f"{col}_change_1h"] = combined[col].diff(1)

    return combined.iloc[[-1]].reset_index(drop=True)


def load_history(path: Path, n_hours: int = 73) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Strip timezone to keep everything tz-naive (consistent with new rows)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp").tail(n_hours).reset_index(drop=True)


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from timestamp column to keep everything tz-naive."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df


def append_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _strip_tz(df)
    if path.exists():
        existing = _strip_tz(pd.read_parquet(path))
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp","city"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Existing: {len(existing):,} → new total: {len(combined):,}")
    else:
        combined = df
        logger.info("Created new live Parquet")
    combined.to_parquet(path, index=False, compression="snappy")


def main():
    logger.info(f"Live feature pipeline — {CITY_NAME} (Open-Meteo, no API key)")

    # 1. Fetch AQ + weather
    df_aq = fetch_current_aq()
    df_wx = fetch_current_weather()

    if df_aq is None:
        logger.error("AQ fetch failed. Exiting.")
        return

    # 2. Align timestamps and merge
    row = df_aq.copy()
    row["city"] = CITY_NAME
    if df_wx is not None:
        for col in WEATHER_COLS:
            row[col] = df_wx[col].iloc[0] if col in df_wx.columns else np.nan
    else:
        logger.warning("Weather unavailable — weather features will be NaN this hour")
        for col in WEATHER_COLS:
            row[col] = np.nan

    # 3. Time features
    row = add_time_features(row)

    # 4. Lag features from history
    history = load_history(OUTPUT_PATH, n_hours=73)
    row = add_lag_features(row, history)

    # 5. Save to Parquet (primary storage + DVC)
    append_to_parquet(row, OUTPUT_PATH)

    # 6. Push to Feast feature store (online + offline)
    try:
        from src.feast_utils import push_to_online_store
        push_to_online_store(row)
    except Exception as e:
        logger.warning(f"Feast push skipped: {e} (Parquet is the fallback)")

    logger.info(f"✅ Done — AQI={row['aqi'].iloc[0]}, "
                f"temp={row.get('temperature', pd.Series([None])).iloc[0]}°C")


if __name__ == "__main__":
    main()
