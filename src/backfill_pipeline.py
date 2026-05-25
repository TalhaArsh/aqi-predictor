"""
AQI Predictor - Phase 2: Historical Backfill (Open-Meteo, AQ + Weather)
========================================================================
Fetches 2 years of historical air quality AND weather data from Open-Meteo.

AQ source:      CAMS-backed us_aqi (EPA multi-pollutant, same as before)
Weather source: ERA5-based archive (temperature, humidity, precipitation,
                wind, pressure, cloud cover)

Both come from the same Open-Meteo API, same timezone (GMT), aligned
hourly timestamps — so the merge is clean and exact.

Output: data/raw/aqi_features_historical.parquet (10 + 6 = 16 raw columns)
All feature engineering (lags, rolling, targets) in impute_features.py.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CITY_NAME = os.getenv("CITY_NAME", "Karachi")
CITY_LAT  = float(os.getenv("CITY_LAT",  "24.8607"))
CITY_LON  = float(os.getenv("CITY_LON",  "67.0011"))

OUTPUT_PATH = Path("data/raw/aqi_features_historical.parquet")

BACKFILL_DAYS      = 730
CHUNK_DAYS         = 30
SLEEP_BETWEEN_CALLS = 0.3

AQ_API   = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_API   = "https://archive-api.open-meteo.com/v1/archive"

AQ_VARS  = ["us_aqi","pm2_5","pm10","nitrogen_dioxide",
            "ozone","carbon_monoxide","sulphur_dioxide","ammonia"]

WX_VARS  = ["temperature_2m","relative_humidity_2m","precipitation",
            "wind_speed_10m","surface_pressure","cloud_cover"]


def fetch_air_quality(lat, lon, start_date, end_date) -> Optional[dict]:
    params = {"latitude": lat, "longitude": lon,
              "hourly": ",".join(AQ_VARS),
              "start_date": start_date, "end_date": end_date,
              "timezone": "GMT"}
    try:
        r = requests.get(AQ_API, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"AQ fetch failed {start_date}→{end_date}: {e}")
        return None


def fetch_weather(lat, lon, start_date, end_date) -> Optional[dict]:
    params = {"latitude": lat, "longitude": lon,
              "hourly": ",".join(WX_VARS),
              "start_date": start_date, "end_date": end_date,
              "timezone": "UTC"}
    try:
        r = requests.get(WX_API, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Weather fetch failed {start_date}→{end_date}: {e}")
        return None


def aq_to_df(data: dict, city: str) -> pd.DataFrame:
    h = data["hourly"]
    df = pd.DataFrame({
        "timestamp":  pd.to_datetime(h["time"]),
        "city":       city,
        "aqi":        h.get("us_aqi"),
        "pm25":       h.get("pm2_5"),
        "pm10":       h.get("pm10"),
        "no2":        h.get("nitrogen_dioxide"),
        "o3":         h.get("ozone"),
        "co":         h.get("carbon_monoxide"),
        "so2":        h.get("sulphur_dioxide"),
        "nh3":        h.get("ammonia"),
    })
    for col in ["aqi","pm25","pm10","no2","o3","co","so2","nh3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def wx_to_df(data: dict) -> pd.DataFrame:
    h = data["hourly"]
    df = pd.DataFrame({
        "timestamp":   pd.to_datetime(h["time"]),
        "temperature": h.get("temperature_2m"),
        "humidity":    h.get("relative_humidity_2m"),
        "precipitation": h.get("precipitation"),
        "wind_speed":  h.get("wind_speed_10m"),
        "pressure":    h.get("surface_pressure"),
        "cloud_cover": h.get("cloud_cover"),
    })
    for col in ["temperature","humidity","precipitation","wind_speed","pressure","cloud_cover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def run_backfill() -> Optional[pd.DataFrame]:
    end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, hour=0)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    logger.info(f"Backfilling {BACKFILL_DAYS} days: {start_dt.date()} → {end_dt.date()}")

    aq_chunks, wx_chunks = [], []
    chunk_start = start_dt
    chunk_num   = 0
    total_chunks = (BACKFILL_DAYS // CHUNK_DAYS) + 1

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_dt)
        chunk_num += 1
        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        # Fetch AQ
        aq_data = fetch_air_quality(CITY_LAT, CITY_LON, s, e)
        if aq_data and aq_data.get("hourly"):
            df_aq = aq_to_df(aq_data, CITY_NAME)
            aq_chunks.append(df_aq)
            logger.info(f"[{chunk_num}/{total_chunks}] AQ  {s}→{e}: "
                        f"{len(df_aq)} rows, "
                        f"AQI valid: {df_aq['aqi'].notna().sum()}")
        else:
            logger.warning(f"[{chunk_num}/{total_chunks}] AQ  {s}→{e}: no data")
        time.sleep(SLEEP_BETWEEN_CALLS)

        # Fetch Weather
        wx_data = fetch_weather(CITY_LAT, CITY_LON, s, e)
        if wx_data and wx_data.get("hourly"):
            df_wx = wx_to_df(wx_data)
            wx_chunks.append(df_wx)
            logger.info(f"[{chunk_num}/{total_chunks}] WX  {s}→{e}: "
                        f"{len(df_wx)} rows, "
                        f"temp valid: {df_wx['temperature'].notna().sum()}")
        else:
            logger.warning(f"[{chunk_num}/{total_chunks}] WX  {s}→{e}: no data")
        time.sleep(SLEEP_BETWEEN_CALLS)

        chunk_start = chunk_end + timedelta(days=1)

    if not aq_chunks:
        logger.error("No AQ data fetched.")
        return None

    # Merge AQ + weather on timestamp
    df_aq = pd.concat(aq_chunks, ignore_index=True)
    df_aq = df_aq.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    if wx_chunks:
        df_wx = pd.concat(wx_chunks, ignore_index=True)
        df_wx = df_wx.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df = pd.merge(df_aq, df_wx, on="timestamp", how="left")
        logger.info(f"Merged AQ + weather: {len(df):,} rows × {len(df.columns)} cols")
    else:
        logger.warning("No weather data — proceeding with AQ only")
        df = df_aq

    # Quality report
    logger.info(f"\nFinal dataset: {len(df):,} rows")
    logger.info(f"Date range:   {df['timestamp'].min()} → {df['timestamp'].max()}")
    for col in ["aqi","pm25","temperature","humidity","precipitation","wind_speed"]:
        if col in df.columns:
            null_pct = df[col].isna().mean()*100
            flag = " ⚠️" if null_pct > 30 else ""
            logger.info(f"  {col:<15} {null_pct:.1f}% null{flag}")

    return df


def main():
    logger.info(f"Starting backfill for {CITY_NAME} (AQ + Weather)")
    df = run_backfill()
    if df is None or df.empty:
        logger.error("No data. Exiting.")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    logger.info(f"✅ Saved {OUTPUT_PATH} ({size_mb:.2f} MB)")
    logger.info("Next: python src/impute_features.py")


if __name__ == "__main__":
    main()
