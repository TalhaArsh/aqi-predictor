# 🌬️ Karachi AQI Predictor — MLOps Pipeline

[![Feature Pipeline](https://github.com/TalhaArsh/aqi-predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/TalhaArsh/aqi-predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/TalhaArsh/aqi-predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/TalhaArsh/aqi-predictor/actions/workflows/training_pipeline.yml)
[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://aqi-predictor-10pearls.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub-orange)](https://dagshub.com/TalhaArsh/aqi-predictor)

> **Live Air Quality Index forecasting for Karachi, Pakistan — 1h to 72h ahead**  
> Built as a full MLOps capstone project for 10Pearls.

🔗 **[Live Dashboard →](https://aqi-predictor-10pearls.streamlit.app/)**  
📊 **[MLflow Experiments →](https://dagshub.com/TalhaArsh/aqi-predictor.mlflow)**  
🗃️ **[DagsHub Repository →](https://dagshub.com/TalhaArsh/aqi-predictor)**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Setup & Installation](#-setup--installation)
- [Running Locally](#-running-locally)
- [Key Findings](#-key-findings)

---

## 🎯 Project Overview

A production-grade MLOps system that:

- **Fetches** live AQI and weather data from Open-Meteo every hour
- **Stores** features in a Feast + Redis Cloud feature store
- **Retrains** 4 ML models (Ridge, Random Forest, CatBoost, XGBoost) daily on a 90-day rolling window
- **Selects** the best model per horizon automatically via MLflow model registry
- **Serves** 6-horizon AQI forecasts (1h, 6h, 12h, 24h, 48h, 72h) through a live Streamlit dashboard

**Why AQI forecasting matters for Karachi:**  
Karachi consistently ranks among the world's most polluted cities. With 15+ million residents, real-time air quality forecasting enables citizens, health workers, and policymakers to make informed decisions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   Open-Meteo CAMS (AQI, PM2.5, PM10, NO2, O3, CO, SO2)        │
│   Open-Meteo Weather (Temp, Humidity, Wind, Pressure, Cloud)    │
└────────────────────┬────────────────────────────────────────────┘
                     │ Every Hour
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE PIPELINE (Hourly)                     │
│   GitHub Actions Cron: "17 * * * *"                            │
│                                                                 │
│   1. Fetch latest AQI + weather from Open-Meteo               │
│   2. Append to live Parquet (GitHub Actions Cache)             │
│   3. Push 99 features to Redis Cloud (TTL: 2h)                │
│   4. feast apply — register feature schema                     │
└────────────────────┬────────────────────────────────────────────┘
                     │ Every Day
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE (Daily)                      │
│   GitHub Actions Cron: "17 2 * * *"                            │
│                                                                 │
│   1. DVC pull historical Parquet from DagsHub                  │
│   2. Restore live Parquet from GitHub Actions Cache            │
│   3. Merge → 17,554 rows (2yr history + live)                  │
│   4. impute_features.py                                         │
│      ├─ Apply 90-day rolling window → ~2,089 rows              │
│      ├─ Feature engineering → 96 features                      │
│      └─ Drop 100%-null columns only (nh3)                      │
│   5. Train 4 models × 6 horizons = 24 models                  │
│      ├─ Ridge (α=100, StandardScaler)                          │
│      ├─ Random Forest (100-300 trees, depth 10-None)           │
│      ├─ CatBoost (categorical: hour/day/month/weekday)         │
│      └─ XGBoost (early stopping, sample weights)               │
│   6. register_best.py                                           │
│      ├─ Compare test RMSE across all families                  │
│      ├─ Archive ALL competing Production models                │
│      └─ Promote winner → MLflow Production stage              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY (DagsHub MLflow)              │
│                                                                 │
│   catboost_1h  → Production  (R²=0.998, RMSE=0.76)            │
│   catboost_6h  → Production  (R²=0.975, RMSE=2.71)            │
│   ridge_12h    → Production  (R²=0.854, RMSE=6.85)            │
│   rf_24h       → Production  (R²=0.665, RMSE=10.92)           │
│   xgboost_48h  → Production  (R²=-0.096, RMSE=15.21)          │
│   rf_72h       → Production  (R²=0.096, RMSE=9.82)            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD (Serving)                      │
│   https://aqi-predictor-10pearls.streamlit.app/                │
│                                                                 │
│   1. Load features from Redis Cloud (live, <2h old)            │
│   2. Discover Production models dynamically from MLflow        │
│   3. Build 96-feature vector (exact column order from scaler)  │
│   4. Predict delta for each horizon                             │
│   5. forecast = current_aqi + predicted_delta                  │
│   6. Display with EPA category color coding                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Data Source** | Open-Meteo CAMS API | Live AQI + weather (free, no API key) |
| **Feature Store** | Feast + Redis Cloud | Online feature serving, TTL=2h |
| **Data Versioning** | DVC + DagsHub | Historical Parquet versioning |
| **Experiment Tracking** | MLflow on DagsHub | Metrics, artifacts, model registry |
| **Models** | Ridge, RF, CatBoost, XGBoost | Multi-family comparison per horizon |
| **CI/CD** | GitHub Actions | Hourly feature + daily training pipelines |
| **Dashboard** | Streamlit Community Cloud | Live forecast serving |
| **Language** | Python 3.11 | All pipeline + training code |

---

## 📊 Model Performance

Models are evaluated on a chronological 80/10/10 train/val/test split using a **90-day rolling window** of Karachi AQI data (Feb–May 2026).

### Best Model Per Horizon (Test Set)

| Horizon | Model | Test RMSE | Test MAE | Test R² | Confidence |
|---|---|---|---|---|---|
| **+1h** | XGBoost | 0.76 | 0.45 | **0.998** | ✅ High |
| **+6h** | CatBoost | 2.71 | 2.02 | **0.975** | ✅ High |
| **+12h** | Ridge | 6.85 | 5.59 | **0.854** | ✅ High |
| **+24h** | RF | 10.92 | 6.62 | **0.665** | ⚠️ Medium |
| **+48h** | XGBoost | 15.21 | 12.53 | -0.096 | ⚠️ Low |
| **+72h** | RF | 9.82 | 7.57 | 0.096 | ⚠️ Low |

### Dataset Statistics

```
Training data:  2,089 rows (90-day rolling window)
Features:       96 (7 pollutants + 6 weather + 9 time + lags/rolling/change)
Target:         aqi_delta_Nh = aqi[t+N] - aqi[t]  (delta framing)
AQI range:      32 – 161 (mean: 80, Moderate category dominates 81.5%)
City:           Karachi, Pakistan (24.86°N, 67.00°E)
```

### Why Delta Framing?

Instead of predicting absolute AQI directly, we predict the **change** in AQI:

```python
target = aqi[t + horizon] - aqi[t]   # predict delta
forecast = current_aqi + predicted_delta   # reconstruct absolute
```

**Advantage:** Tested against direct prediction approach — delta framing yields significantly better R² at all horizons (e.g. +1h: 0.998 vs 0.901 for direct prediction).

---

## 📁 Project Structure

```
aqi-predictor/
│
├── .github/workflows/
│   ├── feature_pipeline.yml      # Hourly: fetch → Redis
│   └── training_pipeline.yml     # Daily: merge → train → register
│
├── feature_store/
│   ├── feature_store.yaml        # Feast config (Redis Cloud online store)
│   └── features.py               # 6 feature views, 96 features defined
│
├── src/
│   ├── feature_pipeline.py       # Hourly live data fetch + Redis push
│   ├── feast_utils.py            # Direct Redis read/write utilities
│   ├── impute_features.py        # Feature engineering (96 features)
│   ├── backfill_pipeline.py      # Historical 2yr data backfill
│   └── training/
│       ├── data_loader.py        # Loads cleaned Parquet for training
│       ├── train_ridge.py        # Ridge regression (6 horizons)
│       ├── train_rf.py           # Random Forest (6 horizons, SHAP)
│       ├── train_catboost.py     # CatBoost (categorical features)
│       ├── train_xgboost.py      # XGBoost (sample weights)
│       └── register_best.py     # Promote best model → MLflow Production
│
├── dashboard/
│   ├── app.py                    # Streamlit dashboard
│   └── requirements.txt          # Pinned dependencies
│
├── data/
│   ├── raw/
│   │   └── aqi_features_historical.parquet   # 2yr history (DVC tracked)
│   └── interim/
│       └── aqi_features_cleaned.parquet      # 90-day engineered features
│
├── check_redis.py                # Local Redis verification script
├── .python-version               # Python 3.11
└── README.md
```

---

## ⚙️ CI/CD Pipeline

### Feature Pipeline — Every Hour

```yaml
Trigger: cron "17 * * * *"

Steps:
  1. Restore live Parquet from GitHub Actions Cache
  2. feast apply (register feature schema to Redis)
  3. python -m src.feature_pipeline
     → Fetch AQI + weather from Open-Meteo
     → Append to live Parquet
     → Push 99 keys to Redis Cloud (TTL=2h)
  4. Save updated live Parquet to cache (live-parquet-v2-{N})
```

### Training Pipeline — Every Day at 02:17 UTC

```yaml
Trigger: cron "17 2 * * *"

Steps:
  1. pip install (Ridge, RF, CatBoost, XGBoost, MLflow, DagsHub)
  2. DVC pull historical Parquet from DagsHub
  3. Restore live Parquet from GitHub Actions Cache
  4. Merge live into historical (column intersection, dedup)
  5. python src/impute_features.py
     → 90-day rolling window filter
     → Feature engineering (lags, rolling means, change features)
     → Exactly 96 features always (drops only 100%-null columns)
  6. Train Ridge  → 6 horizons, logs to MLflow
  7. Train RF     → 6 horizons, SHAP skipped in CI
  8. Train CatBoost → 6 horizons, SHAP skipped in CI
  9. Train XGBoost  → 6 horizons, SHAP skipped in CI
  10. register_best.py
      → Compare test RMSE across all 24 models
      → Archive competing Production models per horizon
      → Promote winner to MLflow Production stage
```

---

## 🔬 Feature Engineering

96 features are engineered from 7 raw pollutant + 6 weather variables:

```
Raw features (13):
  Pollutants: aqi, pm25, pm10, no2, o3, co, so2
  Weather:    temperature, humidity, precipitation, wind_speed, pressure, cloud_cover

Engineered features (83):
  Time:           hour, day, month, day_of_week, is_weekend
                  hour_sin, hour_cos, month_sin, month_cos
  AQI lags:       aqi_lag_1h/3h/6h/12h/24h/48h/72h
  AQI rolling:    aqi_rolling_3h/6h/24h/48h/72h
  AQI change:     aqi_change_1h, aqi_change_3h
  Pollutant lags: pm25/pm10/no2/o3/co/so2_lag_1h/3h/6h/24h
  Weather lags:   temp/humidity/precip/wind/pressure/cloud_lag_1h/6h/12h/24h
  Weather change: temp/humidity/precip/wind/pressure/cloud_change_1h/6h
```

**Critical engineering decision:** The 90-day rolling window captures recent seasonal patterns while avoiding concept drift from older data. Tested against 2-year training: 90-day yields significantly better R² at 1h-24h horizons.

---

## 🔑 Key Findings

### 1. Feature Column Order Is Critical

The most impactful bug found during development: StandardScaler applies scaling by **column position**, not by name. When the dashboard sent features in a different order than training, the scaler applied wrong mean/std to each feature, causing +669 AQI delta predictions.

**Fix:** Read `scaler.feature_names_in_` directly from each model's fitted scaler at inference time and construct the input DataFrame with that exact column order.

### 2. Delta Framing Beats Direct Prediction

| Approach | 1h R² | 6h R² | 12h R² | 24h R² |
|---|---|---|---|---|
| Direct (predict AQI) | 0.901 | 0.719 | 0.595 | 0.213 |
| **Delta (predict change)** | **0.998** | **0.975** | **0.854** | **0.665** |

### 3. 90-Day Window Beats 2-Year Window

Shorter training window captures recent patterns without concept drift:

| Horizon | 2yr R² | 90-day R² |
|---|---|---|
| 1h | 0.985 | **0.998** |
| 6h | 0.803 | **0.975** |
| 12h | 0.764 | **0.854** |

### 4. CatBoost Categorical Feature Handling

Time features (hour, day, month, day_of_week, is_weekend) must be passed as **strings** to CatBoost, not floats. At inference time, a `catboost.Pool` object with `cat_features` indices is required.

### 5. Long-Horizon Limitations

48h and 72h predictions have near-zero or negative R² — AQI 2-3 days ahead is essentially unpredictable from 90 days of data without weather forecast integration. These are shown with reduced opacity (`~`) in the dashboard as a transparency measure.

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11
- Redis Cloud account (free tier: 30MB)
- DagsHub account
- GitHub account

### Clone & Install

```bash
git clone https://github.com/TalhaArsh/aqi-predictor.git
cd aqi-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r dashboard/requirements.txt
pip install feast[redis] dvc[http]
```

### Environment Variables

Create a `.env` file in the project root:

```env
DAGSHUB_USERNAME=TalhaArsh
DAGSHUB_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/TalhaArsh/aqi-predictor.mlflow
REDIS_HOST=your_redis_host
REDIS_PORT=16572
REDIS_PASSWORD=your_redis_password
```

---

## 🏃 Running Locally

```bash
# 1. Fetch latest features (populates Redis)
python -m src.feature_pipeline

# 2. Verify Redis
python check_redis.py

# 3. Run dashboard
streamlit run dashboard/app.py

# 4. Manual training (optional)
python src/impute_features.py
python -m src.training.train_ridge
python -m src.training.train_rf
python -m src.training.train_catboost
python -m src.training.train_xgboost
python -m src.training.register_best
```

---

## 📈 Dashboard Features

- **Live AQI** — current conditions from Redis Cloud (updated hourly)
- **6-horizon forecast** — 1h through 72h with EPA category color coding
- **Hazard alerts** — automatic warnings when AQI forecast ≥ 150
- **Model transparency** — shows which model family is in Production per horizon
- **Confidence indicators** — `~` prefix for low R² horizons (48h, 72h)
- **Fallback** — direct Open-Meteo API fetch if Redis is empty

---

## 👤 Author

**Talha Arsh**  
10Pearls MLOps Capstone Project  
Karachi, Pakistan

---

*Data: Open-Meteo CAMS (free, no API key required)*  
*Deployed: Streamlit Community Cloud*  
*Models: DagsHub MLflow Registry*
