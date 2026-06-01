"""
AQI Predictor Dashboard — Karachi
===================================
Streamlit dashboard for AQI forecasting.

Features:
- Live features from Redis Cloud (Feast online store)
- Production models from DagsHub MLflow registry
- 6-horizon forecast: 1h, 6h, 12h, 24h, 48h, 72h
- EPA category color coding
- Hazard alerts when AQI >= 150

Deploy: Streamlit Community Cloud
  Repo: TalhaArsh/aqi-predictor
  File: dashboard/app.py
  Secrets: DAGSHUB_USERNAME, DAGSHUB_TOKEN,
           MLFLOW_TRACKING_URI, REDIS_PASSWORD,
           REDIS_HOST, REDIS_PORT
"""

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #f9fafb;
}
.stApp { background-color: #0a0f1e; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #38bdf8; }
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-category {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
}
.alert-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.info-box {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #6b7280;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6b7280;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2937;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ── Exact feature order from model scaler ────────────────────
# Retrieved from: model.steps[0][1].feature_names_in_
# This MUST match exactly or predictions will be wrong
RIDGE_FEATURE_COLS = [
    "aqi","pm25","pm10","no2","o3","co","so2",
    "temperature","humidity","precipitation","wind_speed","pressure","cloud_cover",
    "hour","day","month","day_of_week","is_weekend",
    "hour_sin","hour_cos","month_sin","month_cos",
    "aqi_lag_1h","aqi_lag_3h","aqi_lag_6h","aqi_lag_12h",
    "aqi_lag_24h","aqi_lag_48h","aqi_lag_72h",
    "aqi_rolling_3h","aqi_rolling_6h","aqi_rolling_24h",
    "aqi_rolling_48h","aqi_rolling_72h",
    "aqi_change_1h","aqi_change_3h",
    "pm25_lag_1h","pm25_lag_3h",
    "pm10_lag_1h","pm10_lag_3h",
    "no2_lag_1h","no2_lag_3h",
    "o3_lag_1h","o3_lag_3h",
    "co_lag_1h","co_lag_3h",
    "so2_lag_1h","so2_lag_3h",
    "temperature_lag_1h","temperature_lag_6h","temperature_lag_12h","temperature_lag_24h",
    "temperature_change_1h",
    "humidity_lag_1h","humidity_lag_6h","humidity_lag_12h","humidity_lag_24h",
    "humidity_change_1h",
    "precipitation_lag_1h","precipitation_lag_6h","precipitation_lag_12h","precipitation_lag_24h",
    "precipitation_change_1h",
    "wind_speed_lag_1h","wind_speed_lag_6h","wind_speed_lag_12h","wind_speed_lag_24h",
    "wind_speed_change_1h",
    "pressure_lag_1h","pressure_lag_6h","pressure_lag_12h","pressure_lag_24h",
    "pressure_change_1h",
    "cloud_cover_lag_1h","cloud_cover_lag_6h","cloud_cover_lag_12h","cloud_cover_lag_24h",
    "cloud_cover_change_1h",
    "pm25_lag_6h","pm25_lag_24h",
    "pm10_lag_6h","pm10_lag_24h",
    "no2_lag_6h","no2_lag_24h",
    "o3_lag_6h","o3_lag_24h",
    "co_lag_6h","co_lag_24h",
    "so2_lag_6h","so2_lag_24h",
    "temperature_change_6h","humidity_change_6h","precipitation_change_6h",
    "wind_speed_change_6h","pressure_change_6h","cloud_cover_change_6h",
]
assert len(RIDGE_FEATURE_COLS) == 96, f"Expected 96 features, got {len(RIDGE_FEATURE_COLS)}"


# ── AQI helpers ───────────────────────────────────────────────
def get_aqi_category(aqi):
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return "Unknown", "#6b7280"
    aqi = float(aqi)
    if aqi <= 50:   return "Good", "#22c55e"
    if aqi <= 100:  return "Moderate", "#eab308"
    if aqi <= 150:  return "Unhealthy for Sensitive", "#f97316"
    if aqi <= 200:  return "Unhealthy", "#ef4444"
    if aqi <= 300:  return "Very Unhealthy", "#9333ea"
    return "Hazardous", "#7f1d1d"


def get_aqi_advice(aqi):
    if aqi is None: return ""
    aqi = float(aqi)
    if aqi <= 50:   return "Air quality is satisfactory. Outdoor activities are safe."
    if aqi <= 100:  return "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion."
    if aqi <= 150:  return "Members of sensitive groups may experience health effects."
    if aqi <= 200:  return "⚠️ Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exposure."
    if aqi <= 300:  return "🚨 Health alert: everyone may experience serious health effects."
    return "🚨 HAZARDOUS: Health emergency. Avoid all outdoor activities."


# ── Feature loading ───────────────────────────────────────────
@st.cache_data(ttl=300)
def load_features():
    """Load latest features from Redis Cloud."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import redis as redis_lib
        host = os.getenv("REDIS_HOST",
                         "innovative-microquiet-birthday-21764.db.redis.io")
        port = int(os.getenv("REDIS_PORT", "16572"))
        password = os.getenv("REDIS_PASSWORD", "")
        r = redis_lib.Redis(host=host, port=port, password=password,
                            decode_responses=True)
        data = r.hgetall("aqi_predictor:features:Karachi")
        if data:
            features = {}
            for k, v in data.items():
                if k == "_timestamp": continue
                try:
                    features[k] = float(v)
                except (ValueError, TypeError):
                    features[k] = v
            if features.get("aqi"):
                return features, "redis"
    except Exception as e:
        st.warning(f"Redis unavailable: {e}")

    # Fallback: fetch live from Open-Meteo directly
    try:
        import requests
        resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": 24.8607, "longitude": 67.0011,
                "hourly": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
                "timezone": "UTC", "forecast_days": 1,
            }, timeout=10
        )
        data = resp.json()
        now = pd.Timestamp.utcnow().tz_localize(None)
        times = pd.to_datetime(data["hourly"]["time"])
        diffs = [(t - now).total_seconds().__abs__() for t in times]
        idx = diffs.index(min(diffs))
        features = {
            "aqi": data["hourly"]["us_aqi"][idx],
            "pm25": data["hourly"]["pm2_5"][idx],
            "pm10": data["hourly"]["pm10"][idx],
            "no2": data["hourly"]["nitrogen_dioxide"][idx],
            "o3": data["hourly"]["ozone"][idx],
            "co": data["hourly"]["carbon_monoxide"][idx],
            "so2": data["hourly"]["sulphur_dioxide"][idx],
            "hour": now.hour, "day": now.day, "month": now.month,
            "day_of_week": now.dayofweek, "is_weekend": int(now.dayofweek >= 5),
        }
        return features, "direct"
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        return None, "unavailable"


# ── Model loading ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load Production models from DagsHub MLflow."""
    import mlflow
    try:
        import dagshub
        token = os.getenv("DAGSHUB_TOKEN", "")
        user = os.getenv("DAGSHUB_USERNAME", "TalhaArsh")
        if token:
            dagshub.auth.add_app_token(token)
        dagshub.init(repo_owner=user, repo_name="aqi-predictor", mlflow=True)
    except Exception:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "https://dagshub.com/TalhaArsh/aqi-predictor.mlflow"
        )
        mlflow.set_tracking_uri(tracking_uri)

    # Dynamically discover Production models for each horizon
    # Picks the MOST RECENTLY promoted Production model per horizon
    all_model_names = [
        f"{family}_{h}h"
        for family in ["catboost", "xgboost", "rf", "ridge"]
        for h in [1, 6, 12, 24, 48, 72]
    ]
    horizon_model_map = {}   # h → model_name
    horizon_promo_time = {}  # h → creation_timestamp of Production version
    client_temp = mlflow.tracking.MlflowClient()
    for model_name in all_model_names:
        try:
            h = int(model_name.split("_")[-1].replace("h", ""))
            versions = client_temp.search_model_versions(f"name='{model_name}'")
            prod = [v for v in versions if v.current_stage == "Production"]
            if prod:
                # Use creation_timestamp to pick most recently promoted model
                latest_prod = max(prod, key=lambda v: v.creation_timestamp)
                ts = latest_prod.creation_timestamp
                if h not in horizon_promo_time or ts > horizon_promo_time[h]:
                    horizon_promo_time[h] = ts
                    horizon_model_map[h] = model_name
        except Exception:
            pass

    client = mlflow.tracking.MlflowClient()
    models = {}
    model_info = {}

    for h, model_name in horizon_model_map.items():
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            prod = [v for v in versions if v.current_stage == "Production"]
            if not prod:
                st.warning(f"No Production version for {model_name}")
                continue

            latest = max(prod, key=lambda v: int(v.version))
            uri = f"models:/{model_name}/Production"

            # Load model using appropriate loader based on family
            model = None
            try:
                if "catboost" in model_name:
                    try:
                        model = mlflow.catboost.load_model(uri)
                    except Exception:
                        model = mlflow.sklearn.load_model(uri)
                elif "xgboost" in model_name:
                    try:
                        model = mlflow.xgboost.load_model(uri)
                    except Exception:
                        model = mlflow.sklearn.load_model(uri)
                else:
                    # Ridge and RF are sklearn pipelines
                    model = mlflow.sklearn.load_model(uri)
            except Exception as e:
                st.warning(f"Failed to load {model_name}: {e}")
                continue

            if model is not None:
                # Get n_features safely — different models use different attrs
                try:
                    if hasattr(model, "feature_names_") and model.feature_names_:
                        n_features = len(model.feature_names_)
                    elif hasattr(model, "get_n_features_in"):
                        n_features = model.get_n_features_in()
                    elif hasattr(model, "n_features_in_") and model.n_features_in_ > 0:
                        n_features = model.n_features_in_
                    elif hasattr(model, "steps"):
                        last = model.steps[-1][1]
                        n_features = getattr(last, "n_features_in_", 96)
                    else:
                        n_features = 96
                except Exception:
                    n_features = 96
                # Get feature names from scaler for exact column order
                feature_names = None
                if hasattr(model, "steps"):
                    scaler = model.steps[0][1]
                    if hasattr(scaler, "feature_names_in_"):
                        feature_names = list(scaler.feature_names_in_)
                elif hasattr(model, "feature_names_"):
                    feature_names = list(model.feature_names_)

                # Get test_r2 from MLflow run metrics
                test_r2 = None
                test_rmse = None
                try:
                    run_id = latest.run_id
                    if run_id:
                        run = mlflow.get_run(run_id)
                        test_r2 = run.data.metrics.get("test_R2") or run.data.metrics.get("test_r2")
                        test_rmse = run.data.metrics.get("test_RMSE") or run.data.metrics.get("test_rmse")
                except Exception:
                    pass

                models[h] = model
                model_info[h] = {
                    "name": model_name,
                    "n_features": n_features,
                    "version": latest.version,
                    "feature_names": feature_names,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                }

        except Exception as e:
            st.warning(f"Could not load {model_name}: {e}")

    return models, model_info


# ── Feature vector builder ────────────────────────────────────
def build_feature_vector(features: dict, feature_cols: list) -> pd.DataFrame:
    """Build feature DataFrame in exact order model was trained on.

    Uses a pandas DataFrame (not numpy array) so the StandardScaler
    applies scaling using feature names, not position — prevents
    the scaling mismatch that caused +669 delta predictions.

    Missing lag features default to current sensor values rather than 0,
    preventing out-of-distribution inputs to the model.
    """
    current_aqi  = float(features.get("aqi", 69) or 69)
    current_temp = float(features.get("temperature", 29) or 29)
    current_humid= float(features.get("humidity", 60) or 60)
    current_wind = float(features.get("wind_speed", 10) or 10)
    current_press= float(features.get("pressure", 1010) or 1010)
    current_cloud= float(features.get("cloud_cover", 20) or 20)
    current_pm25 = float(features.get("pm25", 15) or 15)
    current_pm10 = float(features.get("pm10", 30) or 30)
    current_no2  = float(features.get("no2", 10) or 10)
    current_o3   = float(features.get("o3", 50) or 50)
    current_co   = float(features.get("co", 0.5) or 0.5)
    current_so2  = float(features.get("so2", 5) or 5)

    # Sensible defaults — current value for lag features,
    # 0 for change/delta features (assume stable)
    sensor_defaults = {
        "aqi": current_aqi, "pm25": current_pm25, "pm10": current_pm10,
        "no2": current_no2, "o3": current_o3, "co": current_co, "so2": current_so2,
        "temperature": current_temp, "humidity": current_humid,
        "precipitation": 0.0, "wind_speed": current_wind,
        "pressure": current_press, "cloud_cover": current_cloud,
    }

    now = datetime.now()
    time_defaults = {
        "hour": float(now.hour), "day": float(now.day),
        "month": float(now.month), "day_of_week": float(now.weekday()),
        "is_weekend": float(now.weekday() >= 5),
        "hour_sin": float(np.sin(2 * np.pi * now.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * now.hour / 24)),
        "month_sin": float(np.sin(2 * np.pi * now.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * now.month / 12)),
    }

    row = {}
    for col in feature_cols:
        val = features.get(col)
        # Use actual value if available and valid
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                row[col] = float(val)
                continue
            except (TypeError, ValueError):
                pass

        # Apply sensible default
        if col in time_defaults:
            row[col] = time_defaults[col]
        elif "change" in col or "rolling" in col:
            row[col] = 0.0  # assume stable
        else:
            # Lag feature — use current sensor value
            prefix = col.split("_lag_")[0].split("_rolling_")[0].split("_change_")[0]
            row[col] = sensor_defaults.get(prefix, 0.0)

    return pd.DataFrame([row])[feature_cols]


# ── Forecast ──────────────────────────────────────────────────
def make_forecasts(features: dict, models: dict, model_info: dict) -> dict:
    """Run models and return absolute AQI forecasts."""
    current_aqi = float(features.get("aqi", 69) or 69)
    forecasts = {}

    for h, model in models.items():
        try:
            n_features = model_info[h]["n_features"]
            model_name = model_info[h].get("name", "")

            # Use model's own feature names for exact column order
            feature_cols = model_info[h].get("feature_names")
            if not feature_cols:
                # Fallback: use standard list trimmed to model's feature count
                feature_cols = RIDGE_FEATURE_COLS if n_features == 96                                else RIDGE_FEATURE_COLS[:n_features]

            X = build_feature_vector(features, feature_cols)

            # Predict based on model family — detected dynamically
            if "catboost" in model_name:
                from catboost import Pool
                cb_feature_names = model_info[h].get("feature_names") or                                    list(model.feature_names_)
                X_cb = build_feature_vector(features, cb_feature_names)
                cat_cols = [c for c in cb_feature_names
                            if c in ["hour","day","month","day_of_week","is_weekend"]]
                for c in cat_cols:
                    X_cb[c] = X_cb[c].astype(int).astype(str)
                cat_indices = [X_cb.columns.tolist().index(c) for c in cat_cols]
                pool = Pool(X_cb, cat_features=cat_indices)
                pred = model.predict(pool)
            elif "xgboost" in model_name:
                # XGBoost needs numpy array
                try:
                    pred = model.predict(X.values)
                except Exception:
                    pred = model.predict(X)
            else:
                # Ridge and RF — sklearn Pipeline with named DataFrame
                try:
                    pred = model.predict(X)
                except Exception as e:
                    forecasts[h] = None
                    continue

            # Handle different output shapes
            if hasattr(pred, "__len__"):
                delta = float(pred.flat[0] if hasattr(pred, "flat") else pred[0])
            else:
                delta = float(pred)

            # Hard sanity cap — AQI dataset range is 32-161
            # Max possible delta in any horizon = 161 - 32 = 129
            # If delta exceeds this, features were garbage (e.g. Redis empty)
            MAX_AQI_RANGE = 130
            if abs(delta) > MAX_AQI_RANGE:
                # Features were likely all zeros/defaults — prediction unusable
                forecasts[h] = None
                continue
            # Soft cap per horizon
            max_delta = {1: 15, 6: 30, 12: 50, 24: 70, 48: 90, 72: 100}
            max_d = max_delta.get(h, 80)
            if abs(delta) > max_d:
                delta = float(np.clip(delta, -max_d, max_d))

            forecast = float(np.clip(current_aqi + delta, 0, 500))
            forecasts[h] = round(forecast, 1)
        except Exception as e:
            forecasts[h] = None

    return forecasts


# ── Main dashboard ────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
        <div style="font-family:'Space Mono',monospace;font-size:28px;font-weight:700;">
            🌬️ KARACHI AQI FORECAST
        </div>
        <div style="font-family:'DM Sans';font-size:13px;color:#6b7280;margin-top:4px;">
            <span class="status-dot"></span>Live · Open-Meteo + MLflow
        </div>
    </div>
    <div style="color:#6b7280;font-size:14px;margin-bottom:24px;">
        Air Quality Index predictions for the next 72 hours
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Fetching live features..."):
        features, source = load_features()

    if features is None:
        st.error("Unable to load features. Check Redis/API connection.")
        return

    with st.spinner("Loading models from DagsHub MLflow..."):
        models, model_info = load_models()

    if not models:
        st.error("No models loaded. Check DagsHub MLflow connection.")
        return

    # Make forecasts
    forecasts = make_forecasts(features, models, model_info)

    # Current conditions
    current_aqi  = features.get("aqi")
    current_temp = features.get("temperature")
    current_wind = features.get("wind_speed")
    current_humid= features.get("humidity")
    cat, color   = get_aqi_category(current_aqi)

    st.markdown('<div class="section-header">Current Conditions</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AQI Now</div>
            <div class="metric-value" style="color:{color}">
                {int(current_aqi) if current_aqi else "—"}
            </div>
            <span class="metric-category" style="background:{color}22;color:{color}">
                {cat}
            </span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Temperature</div>
            <div class="metric-value" style="color:#38bdf8">
                {f"{current_temp:.0f}°C" if current_temp else "—"}
            </div>
            <span class="metric-category" style="background:#38bdf822;color:#38bdf8">
                Open-Meteo
            </span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Wind Speed</div>
            <div class="metric-value" style="color:#a78bfa">
                {f"{current_wind:.1f}" if current_wind else "—"}
            </div>
            <span class="metric-category" style="background:#a78bfa22;color:#a78bfa">
                km/h
            </span>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Humidity</div>
            <div class="metric-value" style="color:#34d399">
                {f"{current_humid:.0f}%" if current_humid else "—"}
            </div>
            <span class="metric-category" style="background:#34d39922;color:#34d399">
                Relative
            </span>
        </div>""", unsafe_allow_html=True)

    # Health advice
    advice = get_aqi_advice(current_aqi)
    if advice:
        is_alert = current_aqi and float(current_aqi) > 150
        box_class = "alert-box" if is_alert else "info-box"
        st.markdown(f'<div class="{box_class}">{advice}</div>',
                    unsafe_allow_html=True)

    # Feature quality warning
    n_redis_features = sum(1 for k, v in features.items()
                           if v is not None and k != "_timestamp")
    has_lag_history = features.get("aqi_lag_24h") is not None
    if n_redis_features < 50:
        st.info(
            f"ℹ️ Building feature history ({n_redis_features}/96 features available). "
            "Lag features populate automatically — fully accurate after 72h of pipeline runs. "
            "1h and 6h forecasts are reliable now."
        )

    # Hazard alerts
    hazard = {h: v for h, v in forecasts.items()
              if v is not None and v >= 150}
    if hazard:
        st.markdown('<div class="section-header">⚠️ Hazard Alerts</div>',
                    unsafe_allow_html=True)
        for h, v in hazard.items():
            cat_h, color_h = get_aqi_category(v)
            st.markdown(f"""
            <div class="alert-box">
                <span style="font-size:20px">🚨</span>
                <div>
                    <strong style="color:{color_h}">
                        {cat_h} forecast in {h} hours
                    </strong>
                    <div style="font-size:13px;color:#9ca3af;margin-top:2px;">
                        Predicted AQI: {v:.0f}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Forecast metric cards
    st.markdown('<div class="section-header">72-Hour Forecast</div>',
                unsafe_allow_html=True)

    labels = {1:"1 Hour",6:"6 Hours",12:"12 Hours",
              24:"24 Hours",48:"48 Hours",72:"72 Hours"}
    # Models with negative R² are unreliable — show with caveat
    r2_map_actual = {1:0.999, 6:0.966, 12:0.914, 24:0.618, 48:-0.105, 72:-0.824}
    cols = st.columns(6)
    for i, h in enumerate([1,6,12,24,48,72]):
        val = forecasts.get(h)
        r2 = r2_map_actual.get(h, 0)
        # Hide unreliable predictions when insufficient feature history
        if r2 < 0 and n_redis_features < 50:
            val = None
        cat_f, color_f = get_aqi_category(val)
        # Dim negative R² models
        opacity = "1.0" if r2 >= 0 else "0.6"
        suffix = "" if r2 >= 0 else "~"  # ~ means approximate
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="opacity:{opacity}">
                <div class="metric-label">{labels[h]}</div>
                <div class="metric-value" style="color:{color_f}">
                    {f"{suffix}{val:.0f}" if val is not None else "—"}
                </div>
                <span class="metric-category"
                      style="background:{color_f}22;color:{color_f}">
                    {cat_f}
                </span>
            </div>""", unsafe_allow_html=True)

    # Forecast chart
    st.markdown('<div class="section-header">Forecast Timeline</div>',
                unsafe_allow_html=True)

    horizons = [0, 1, 6, 12, 24, 48, 72]
    values   = [current_aqi] + [forecasts.get(h) for h in [1,6,12,24,48,72]]
    valid_h  = [h for h, v in zip(horizons, values) if v is not None]
    valid_v  = [v for v in values if v is not None]
    point_colors = [get_aqi_category(v)[1] for v in valid_v]

    fig = go.Figure()
    for lo, hi, col_b, name in [
        (0,50,"rgba(34,197,94,0.06)","Good"),
        (50,100,"rgba(234,179,8,0.06)","Moderate"),
        (100,150,"rgba(249,115,22,0.06)","Unhealthy Sensitive"),
        (150,200,"rgba(239,68,68,0.06)","Unhealthy"),
        (200,300,"rgba(147,51,234,0.06)","Very Unhealthy"),
    ]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col_b, layer="below",
                      line_width=0, annotation_text=name,
                      annotation_position="right",
                      annotation_font_size=10,
                      annotation_font_color="#374151")

    fig.add_trace(go.Scatter(
        x=valid_h, y=valid_v,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=10, color=point_colors,
                    line=dict(color="#0a0f1e", width=2)),
        hovertemplate="<b>+%{x}h</b><br>AQI: %{y:.0f}<extra></extra>",
    ))
    fig.add_hline(y=150, line_dash="dash",
                  line_color="rgba(239,68,68,0.5)",
                  annotation_text="Unhealthy threshold (150)",
                  annotation_font_color="#ef4444",
                  annotation_font_size=11)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        font=dict(family="DM Sans", color="#9ca3af"),
        xaxis=dict(
            title="Hours from now",
            tickvals=[0,1,6,12,24,48,72],
            ticktext=["Now","+1h","+6h","+12h","+24h","+48h","+72h"],
            gridcolor="rgba(31,41,55,0.8)", zeroline=False,
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="rgba(31,41,55,0.8)", zeroline=False,
            range=[0, max(max(v for v in valid_v if v), 200) * 1.15],
        ),
        margin=dict(l=0, r=120, t=20, b=0),
        height=380, showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model info
    st.markdown('<div class="section-header">Model Information</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        # Dynamic model info from MLflow — reflects actual Production models
        r2_map = {1:0.999, 6:0.966, 12:0.914, 24:0.618, 48:-0.105, 72:-0.824}
        # Replace the table building block:
        rows = []
        for h in [1,6,12,24,48,72]:
            info = model_info.get(h, {})
            model_name = info.get("name", "—").replace(f"_{h}h", "").upper()
            live_r2 = info.get("test_r2") or FALLBACK_R2.get(h, 0)
            live_rmse = info.get("test_rmse")  # pull from MLflow
            rows.append({
                "Horizon": f"+{h}h",
                "Model": model_name,
                "Features": str(info.get("n_features", "—")),
                "Test RMSE": f"±{live_rmse:.1f}" if live_rmse else "—",
                "Test R²": f"{live_r2:.3f}",
                "Forecast AQI": f"{forecasts.get(h):.0f}" if forecasts.get(h) else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col_b:
        now = datetime.now(timezone.utc)
        st.markdown(f"""
        <div class="info-box">
            <strong>Data Source</strong><br>
            AQ + Weather: Open-Meteo CAMS (free, no API key)<br>
            Feature Store: Feast + Redis Cloud<br>
            Model Registry: DagsHub MLflow<br>
            Feature Source: {source.upper()}
        </div>
        <div class="info-box" style="margin-top:8px;">
            <strong>Last Updated</strong><br>
            {now.strftime('%Y-%m-%d %H:%M UTC')}
        </div>
        <div class="info-box" style="margin-top:8px;">
            <strong>Coverage</strong><br>
            City: Karachi, Pakistan (24.86°N, 67.00°E)<br>
            Training: 90-day rolling window, retrained daily
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:48px;padding-top:24px;
                border-top:1px solid #1f2937;color:#374151;font-size:12px;">
        AQI Predictor · 10Pearls Capstone · Karachi Air Quality Forecasting
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()