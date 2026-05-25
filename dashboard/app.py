"""
AQI Predictor Dashboard — Phase 5
===================================
Streamlit dashboard for Karachi AQI forecasting.

Architecture:
  - Reads live features from Feast online store (Redis Cloud)
  - Loads Production models from DagsHub MLflow registry
  - Shows 6-horizon forecast: 1h, 6h, 12h, 24h, 48h, 72h
  - Hazard alerts when AQI >= 150
  - Metric cards + line chart

Deploy: Streamlit Community Cloud
  - Connect to GitHub repo: TalhaArsh/aqi-predictor
  - Main file: dashboard/app.py
  - Add secrets: DAGSHUB_USERNAME, DAGSHUB_TOKEN,
                 MLFLOW_TRACKING_URI, REDIS_PASSWORD
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

:root {
    --good:        #22c55e;
    --moderate:    #eab308;
    --sensitive:   #f97316;
    --unhealthy:   #ef4444;
    --very-unhealthy: #9333ea;
    --hazardous:   #7f1d1d;
    --bg:          #0a0f1e;
    --card:        #111827;
    --border:      #1f2937;
    --text:        #f9fafb;
    --muted:       #6b7280;
    --accent:      #38bdf8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
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
    color: var(--muted);
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--good);
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ── AQI helpers ───────────────────────────────────────────────
def get_aqi_category(aqi):
    if aqi is None or np.isnan(aqi):
        return "Unknown", "#6b7280"
    aqi = float(aqi)
    if aqi <= 50:   return "Good", "#22c55e"
    if aqi <= 100:  return "Moderate", "#eab308"
    if aqi <= 150:  return "Unhealthy for Sensitive", "#f97316"
    if aqi <= 200:  return "Unhealthy", "#ef4444"
    if aqi <= 300:  return "Very Unhealthy", "#9333ea"
    return "Hazardous", "#7f1d1d"


def get_aqi_advice(aqi):
    if aqi is None or np.isnan(aqi): return ""
    aqi = float(aqi)
    if aqi <= 50:   return "Air quality is satisfactory. Outdoor activities are safe."
    if aqi <= 100:  return "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion."
    if aqi <= 150:  return "Members of sensitive groups may experience health effects. General public is not likely to be affected."
    if aqi <= 200:  return "⚠️ Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exposure."
    if aqi <= 300:  return "🚨 Health alert: everyone may experience serious health effects. Avoid outdoor activities."
    return "🚨 HAZARDOUS: Health emergency. Everyone should avoid all outdoor activities."


# ── Feature loading ───────────────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes
def load_features():
    """Load latest features from Feast online store (Redis Cloud)."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.feast_utils import get_online_features
        features = get_online_features("Karachi")
        if features and features.get("aqi") is not None:
            return features, "feast"
    except Exception as e:
        st.warning(f"Feast unavailable: {e}")

    # Fallback: read from live Parquet
    try:
        live_path = Path(__file__).parent.parent / "data/raw/aqi_features_live.parquet"
        if live_path.exists():
            df = pd.read_parquet(live_path)
            latest = df.sort_values("timestamp").iloc[-1].to_dict()
            return latest, "parquet"
    except Exception:
        pass

    return None, "unavailable"


# ── Model loading ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load Production models from DagsHub MLflow registry."""
    import dagshub
    import mlflow

    dagshub_user = os.getenv("DAGSHUB_USERNAME", "TalhaArsh")
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"https://dagshub.com/{dagshub_user}/aqi-predictor.mlflow"
    )

    try:
        if dagshub_token:
            dagshub.auth.add_app_token(dagshub_token)
        dagshub.init(repo_owner=dagshub_user,
                     repo_name="aqi-predictor", mlflow=True)
        mlflow.set_tracking_uri(tracking_uri)
    except Exception:
        mlflow.set_tracking_uri(tracking_uri)

    models = {}
    horizon_model_map = {
        1:  ("rf",    "rf_1h"),
        6:  ("rf",    "rf_6h"),
        12: ("ridge", "ridge_12h"),
        24: ("ridge", "ridge_24h"),
        48: ("ridge", "ridge_48h"),
        72: ("ridge", "ridge_72h"),
    }

    client = mlflow.tracking.MlflowClient()
    for h, (family, model_name) in horizon_model_map.items():
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            prod = [v for v in versions if v.current_stage == "Production"]
            if prod:
                latest = max(prod, key=lambda v: int(v.version))
                uri = f"models:/{model_name}/Production"
                models[h] = mlflow.sklearn.load_model(uri)
        except Exception as e:
            st.warning(f"Could not load {model_name}: {e}")

    return models


# ── Feature vector builder ────────────────────────────────────
FEATURE_COLS = [
    "aqi","pm25","pm10","no2","o3","co","so2",
    "temperature","humidity","precipitation","wind_speed","pressure","cloud_cover",
    "hour","day","month","day_of_week","is_weekend",
    "hour_sin","hour_cos","month_sin","month_cos",
    "aqi_lag_1h","aqi_lag_3h","aqi_lag_6h","aqi_lag_12h",
    "aqi_lag_24h","aqi_lag_48h","aqi_lag_72h",
    "aqi_rolling_3h","aqi_rolling_6h","aqi_rolling_24h",
    "aqi_rolling_48h","aqi_rolling_72h",
    "aqi_change_1h","aqi_change_3h",
] + [
    f"{p}_lag_{h}h"
    for p in ["pm25","pm10","no2","o3","co","so2"]
    for h in [1,3,6,24]
] + [
    f"{col}_lag_{h}h"
    for col in ["temperature","humidity","precipitation","wind_speed","pressure","cloud_cover"]
    for h in [1,6,12,24]
] + [
    f"{col}_change_{h}h"
    for col in ["temperature","humidity","precipitation","wind_speed","pressure","cloud_cover"]
    for h in [1,6]
]


def build_feature_vector(features: dict) -> np.ndarray:
    row = []
    for col in FEATURE_COLS:
        val = features.get(col, np.nan)
        try:
            row.append(float(val) if val is not None else np.nan)
        except (TypeError, ValueError):
            row.append(np.nan)
    arr = np.array(row).reshape(1, -1)
    # Impute NaN with 0 (models handle this gracefully)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


# ── Forecast ──────────────────────────────────────────────────
def make_forecasts(features: dict, models: dict) -> dict:
    """Run all 6 models and return absolute AQI forecasts."""
    current_aqi = float(features.get("aqi", 0) or 0)
    X = build_feature_vector(features)
    forecasts = {}
    for h, model in models.items():
        try:
            delta = float(model.predict(X)[0])
            forecast = float(np.clip(current_aqi + delta, 0, 500))
            forecasts[h] = round(forecast, 1)
        except Exception as e:
            forecasts[h] = None
    return forecasts


# ── Main dashboard ────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
        <div style="font-family:'Space Mono',monospace; font-size:28px; font-weight:700;">
            🌬️ KARACHI AQI FORECAST
        </div>
        <div style="font-family:'DM Sans'; font-size:13px; color:#6b7280; margin-top:4px;">
            <span class="status-dot"></span>Live · Powered by Open-Meteo + MLflow
        </div>
    </div>
    <div style="color:#6b7280; font-size:14px; margin-bottom:24px;">
        Air Quality Index predictions for the next 72 hours
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Fetching live features..."):
        features, source = load_features()

    if features is None:
        st.error("Unable to load features. Check Feast/Redis connection.")
        return

    with st.spinner("Loading models..."):
        models = load_models()

    if not models:
        st.error("No models loaded. Check DagsHub MLflow connection.")
        return

    # Make forecasts
    forecasts = make_forecasts(features, models)

    # Current conditions row
    current_aqi = features.get("aqi")
    current_temp = features.get("temperature")
    current_wind = features.get("wind_speed")
    current_humid = features.get("humidity")

    cat, color = get_aqi_category(current_aqi)

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
        is_alert = current_aqi and current_aqi > 150
        box_class = "alert-box" if is_alert else "info-box"
        st.markdown(f'<div class="{box_class}">{advice}</div>',
                    unsafe_allow_html=True)

    # Hazard alerts
    hazard_horizons = {h: v for h, v in forecasts.items()
                       if v is not None and v >= 150}
    if hazard_horizons:
        st.markdown('<div class="section-header">⚠️ Hazard Alerts</div>',
                    unsafe_allow_html=True)
        for h, v in hazard_horizons.items():
            cat_h, color_h = get_aqi_category(v)
            st.markdown(f"""
            <div class="alert-box">
                <span style="font-size:20px">🚨</span>
                <div>
                    <strong style="color:{color_h}">
                        {cat_h} conditions forecast in {h} hours
                    </strong>
                    <div style="font-size:13px;color:#9ca3af;margin-top:2px;">
                        Predicted AQI: {v:.0f} — Consider limiting outdoor activities
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Forecast metric cards
    st.markdown('<div class="section-header">72-Hour Forecast</div>',
                unsafe_allow_html=True)

    horizon_labels = {1: "1 Hour", 6: "6 Hours", 12: "12 Hours",
                      24: "24 Hours", 48: "48 Hours", 72: "72 Hours"}
    cols = st.columns(6)
    for i, h in enumerate([1, 6, 12, 24, 48, 72]):
        val = forecasts.get(h)
        cat_f, color_f = get_aqi_category(val)
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{horizon_labels[h]}</div>
                <div class="metric-value" style="color:{color_f}">
                    {f"{val:.0f}" if val is not None else "—"}
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
    values = [current_aqi] + [forecasts.get(h) for h in [1, 6, 12, 24, 48, 72]]

    # AQI band colors
    fig = go.Figure()

    # Background AQI bands
    bands = [
        (0,   50,  "rgba(34,197,94,0.06)",   "Good"),
        (50,  100, "rgba(234,179,8,0.06)",    "Moderate"),
        (100, 150, "rgba(249,115,22,0.06)",   "Unhealthy Sensitive"),
        (150, 200, "rgba(239,68,68,0.06)",    "Unhealthy"),
        (200, 300, "rgba(147,51,234,0.06)",   "Very Unhealthy"),
    ]
    for lo, hi, color_band, name in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color_band,
                      layer="below", line_width=0,
                      annotation_text=name,
                      annotation_position="right",
                      annotation_font_size=10,
                      annotation_font_color="#374151")

    # Forecast line
    valid_h = [h for h, v in zip(horizons, values) if v is not None]
    valid_v = [v for v in values if v is not None]
    point_colors = [get_aqi_category(v)[1] for v in valid_v]

    fig.add_trace(go.Scatter(
        x=valid_h, y=valid_v,
        mode="lines+markers",
        name="AQI Forecast",
        line=dict(color="#38bdf8", width=2.5, dash="solid"),
        marker=dict(size=10, color=point_colors,
                    line=dict(color="#0a0f1e", width=2)),
        hovertemplate="<b>+%{x}h</b><br>AQI: %{y:.0f}<extra></extra>",
    ))

    # Hazard threshold line
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
            tickvals=[0, 1, 6, 12, 24, 48, 72],
            ticktext=["Now", "+1h", "+6h", "+12h", "+24h", "+48h", "+72h"],
            gridcolor="rgba(31,41,55,0.8)",
            zeroline=False,
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="rgba(31,41,55,0.8)",
            zeroline=False,
            range=[0, max(max(v for v in valid_v if v), 200) * 1.15],
        ),
        margin=dict(l=0, r=120, t=20, b=0),
        height=380,
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Model info + metadata
    st.markdown('<div class="section-header">Model Information</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        model_map = {1:"RF",6:"RF",12:"Ridge",24:"Ridge",48:"Ridge",72:"Ridge"}
        r2_map = {1:0.993,6:0.892,12:0.764,24:0.525,48:0.136,72:0.024}
        rows = []
        for h in [1,6,12,24,48,72]:
            rows.append({
                "Horizon": f"+{h}h",
                "Model": model_map[h],
                "Test R²": f"{r2_map[h]:.3f}",
                "Forecast AQI": f"{forecasts.get(h):.0f}" if forecasts.get(h) else "—"
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
            Training data: May 2024 – May 2026 (17,472 hourly obs.)
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:48px;padding-top:24px;
                border-top:1px solid #1f2937;color:#374151;font-size:12px;">
        AQI Predictor · 10Pearls Capstone Project ·
        Karachi Air Quality Forecasting Pipeline
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
