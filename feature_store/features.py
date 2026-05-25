"""
AQI Predictor — Feast Feature Definitions
==========================================
Defines all feature views, entities, and data sources.

Architecture:
  Offline store: Parquet files (data/interim/aqi_features_cleaned.parquet)
    → Used by training pipeline for point-in-time correct feature retrieval
    → Prevents data leakage: only features available at prediction time are used

  Online store: SQLite (data/feast/online_store.db)
    → Used by feature_pipeline.py to write latest features each hour
    → Used by dashboard for real-time inference (millisecond lookup)

Why this prevents training-serving skew:
  Both training (get_historical_features) and serving (get_online_features)
  read from the same Feast feature definitions. If a feature name or
  computation changes, it must be updated here — one source of truth.

Feature groups:
  aqi_features:       Current AQI + pollutants
  weather_features:   Current weather conditions
  aqi_lag_features:   AQI lag/rolling/change history
  pollutant_lag_features: Per-pollutant lag history
  weather_lag_features:   Weather lag + change history
  time_features:      Cyclical time encodings
"""

from datetime import timedelta
from feast import (
    Entity, FeatureView, Field, FileSource,
    ValueType, PushSource,
)
from feast.types import Float32, Int32

# ─────────────────────── Entity ───────────────────────────────
# Entity = what we're making predictions for
# Here: a city at a specific timestamp

city = Entity(
    name="city",
    description="City name — identifies the monitoring location",
    value_type=ValueType.STRING,
)

# ─────────────────────── Data Sources ─────────────────────────

aqi_source = FileSource(
    path="data/interim/aqi_features_cleaned.parquet",
    timestamp_field="timestamp",
    description="Cleaned hourly AQI + weather features from Open-Meteo",
)

# PushSource for online store — used by feature_pipeline.py each hour
aqi_push_source = PushSource(
    name="aqi_push_source",
    batch_source=aqi_source,
)

# ─────────────────────── Feature Views ────────────────────────

aqi_features = FeatureView(
    name="aqi_features",
    entities=[city],
    ttl=timedelta(hours=2),   # online store: expire after 2h if not updated
    schema=[
        Field(name="aqi",   dtype=Float32, description="US AQI (EPA multi-pollutant)"),
        Field(name="pm25",  dtype=Float32, description="PM2.5 μg/m³"),
        Field(name="pm10",  dtype=Float32, description="PM10 μg/m³"),
        Field(name="no2",   dtype=Float32, description="NO2 μg/m³"),
        Field(name="o3",    dtype=Float32, description="O3 μg/m³"),
        Field(name="co",    dtype=Float32, description="CO μg/m³"),
        Field(name="so2",   dtype=Float32, description="SO2 μg/m³"),
    ],
    source=aqi_push_source,
    online=True,
    description="Current AQI and pollutant readings",
)

weather_features = FeatureView(
    name="weather_features",
    entities=[city],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="temperature",  dtype=Float32, description="Temperature °C"),
        Field(name="humidity",     dtype=Float32, description="Relative humidity %"),
        Field(name="precipitation",dtype=Float32, description="Precipitation mm"),
        Field(name="wind_speed",   dtype=Float32, description="Wind speed km/h"),
        Field(name="pressure",     dtype=Float32, description="Surface pressure hPa"),
        Field(name="cloud_cover",  dtype=Float32, description="Cloud cover %"),
    ],
    source=aqi_push_source,
    online=True,
    description="Current weather conditions from Open-Meteo",
)

time_features = FeatureView(
    name="time_features",
    entities=[city],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="hour",        dtype=Int32,   description="Hour of day (0-23)"),
        Field(name="day",         dtype=Int32,   description="Day of month"),
        Field(name="month",       dtype=Int32,   description="Month (1-12)"),
        Field(name="day_of_week", dtype=Int32,   description="Day of week (0=Mon)"),
        Field(name="is_weekend",  dtype=Int32,   description="1 if weekend"),
        Field(name="hour_sin",    dtype=Float32, description="Cyclical hour encoding (sin)"),
        Field(name="hour_cos",    dtype=Float32, description="Cyclical hour encoding (cos)"),
        Field(name="month_sin",   dtype=Float32, description="Cyclical month encoding (sin)"),
        Field(name="month_cos",   dtype=Float32, description="Cyclical month encoding (cos)"),
    ],
    source=aqi_push_source,
    online=True,
    description="Cyclical time features",
)

aqi_lag_features = FeatureView(
    name="aqi_lag_features",
    entities=[city],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="aqi_lag_1h",      dtype=Float32),
        Field(name="aqi_lag_3h",      dtype=Float32),
        Field(name="aqi_lag_6h",      dtype=Float32),
        Field(name="aqi_lag_12h",     dtype=Float32),
        Field(name="aqi_lag_24h",     dtype=Float32),
        Field(name="aqi_lag_48h",     dtype=Float32),
        Field(name="aqi_lag_72h",     dtype=Float32),
        Field(name="aqi_rolling_3h",  dtype=Float32),
        Field(name="aqi_rolling_6h",  dtype=Float32),
        Field(name="aqi_rolling_24h", dtype=Float32),
        Field(name="aqi_rolling_48h", dtype=Float32),
        Field(name="aqi_rolling_72h", dtype=Float32),
        Field(name="aqi_change_1h",   dtype=Float32),
        Field(name="aqi_change_3h",   dtype=Float32),
    ],
    source=aqi_push_source,
    online=True,
    description="AQI lag, rolling window, and change features",
)

pollutant_lag_features = FeatureView(
    name="pollutant_lag_features",
    entities=[city],
    ttl=timedelta(hours=2),
    schema=[
        Field(name=f"{p}_lag_{h}h", dtype=Float32,
              description=f"{p.upper()} {h}h lag")
        for p in ["pm25","pm10","no2","o3","co","so2"]
        for h in [1, 3, 6, 24]
    ],
    source=aqi_push_source,
    online=True,
    description="Per-pollutant lag features (no rolling — avoids multicollinearity)",
)

weather_lag_features = FeatureView(
    name="weather_lag_features",
    entities=[city],
    ttl=timedelta(hours=2),
    schema=[
        *[Field(name=f"{col}_lag_{h}h", dtype=Float32)
          for col in ["temperature","humidity","precipitation",
                      "wind_speed","pressure","cloud_cover"]
          for h in [1, 6, 12, 24]],
        *[Field(name=f"{col}_change_{h}h", dtype=Float32)
          for col in ["temperature","humidity","precipitation",
                      "wind_speed","pressure","cloud_cover"]
          for h in [1, 6]],
    ],
    source=aqi_push_source,
    online=True,
    description="Weather lag and change features (key for 48h/72h prediction)",
)
