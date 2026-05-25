# CI/CD Setup Guide

## Overview

Two GitHub Actions workflows manage the live production pipeline:

| Workflow | Schedule | What it does |
|---|---|---|
| Feature Pipeline | Every hour | Fetch live AQI from Open-Meteo → append Parquet → push DagsHub |
| Training Pipeline | Daily 02:00 UTC | Merge data → impute → retrain Ridge/RF/CatBoost → register best |

---

## Important: No AQICN or OpenWeather API keys needed

The live pipeline was rewritten to use **Open-Meteo** — the same source
as the training data. This eliminates training-serving skew (a common
production bug where live data comes from a different distribution than
what the model was trained on).

Open-Meteo is free and requires no API key.

---

## Step 1: One-time DVC setup (run locally in PowerShell)

```powershell
# Make sure your .env is loaded first
Get-Content .env | ForEach-Object {
    if ($_ -match "^([^#][^=]+)=(.+)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# Initialize DVC
dvc init

# Add DagsHub remote
dvc remote add -d dagshub https://dagshub.com/TalhaArsh/aqi-predictor.dvc

# Configure credentials (stored locally only, never committed)
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user $env:DAGSHUB_USERNAME
dvc remote modify dagshub --local password $env:DAGSHUB_TOKEN

# Track your data files
dvc add data/raw/aqi_features_historical.parquet
dvc add data/raw/aqi_features_live.parquet
dvc add data/interim/aqi_features_cleaned.parquet

# Push data to DagsHub
dvc push

# Commit everything to git
git add .dvc/config dvc.yaml .gitignore .github/
git add data/raw/aqi_features_historical.parquet.dvc
git add data/raw/aqi_features_live.parquet.dvc
git add data/interim/aqi_features_cleaned.parquet.dvc
git commit -m "feat: add DVC data versioning + GitHub Actions CI/CD"
git push origin main
```

---

## Step 2: Add GitHub Secrets

Go to: GitHub repo → Settings → Secrets and variables → Actions → New repository secret

Only **3 secrets needed** (no AQICN or OpenWeather keys):

| Secret name | Value |
|---|---|
| `DAGSHUB_USERNAME` | `TalhaArsh` |
| `DAGSHUB_TOKEN` | DagsHub → Settings → Tokens → Generate new token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/TalhaArsh/aqi-predictor.mlflow` |

---

## Step 3: Verify workflows are active

After pushing to `main`, go to:
- GitHub repo → Actions tab → you should see both workflows listed
- Trigger manually: Actions → Feature Pipeline → Run workflow
- Check it goes green ✅

---

## Data flow

```
Every hour:
  GitHub Actions
    ├─ dvc pull  (get current live Parquet from DagsHub)
    ├─ Open-Meteo API  (no key, same source as training)
    ├─ append row to Parquet
    ├─ dvc push  (updated Parquet → DagsHub)
    └─ git push  (.dvc pointer file)

Every day 02:00 UTC:
  GitHub Actions
    ├─ dvc pull  (historical + live Parquet)
    ├─ merge live → historical
    ├─ impute_features.py  (lags, rolling, targets)
    ├─ train_ridge.py
    ├─ train_rf.py
    ├─ train_catboost.py
    ├─ register_best.py  (promote winner → Production)
    ├─ dvc push  (updated cleaned Parquet)
    └─ git push  (.dvc pointers)
```

---

## Troubleshooting

**DVC push fails with 401 Unauthorized**
→ Regenerate DagsHub token, update GitHub secret + local config

**Training workflow times out (>90 min)**
→ Reduce CatBoost `iterations` in `train_catboost.py` for CI runs

**`[skip ci]` in commit messages**
→ Intentional — prevents CI commit from triggering another CI run

**Open-Meteo returns null AQI for current hour**
→ CAMS has ~1h latency. Pipeline falls back to last non-null row automatically.
