# ============================================================
# DVC Setup Script for Windows PowerShell
# Run this ONCE from your project root to initialize DVC
# and connect it to DagsHub.
# ============================================================

Write-Host "Setting up DVC with DagsHub remote..." -ForegroundColor Cyan

# 1. Initialize DVC (creates .dvc/ directory)
dvc init

# 2. Add DagsHub as the remote storage backend
dvc remote add -d dagshub https://dagshub.com/TalhaArsh/aqi-predictor.dvc

# 3. Configure credentials (reads from .env automatically via DVC)
# These are stored locally only — never committed to git
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user $env:DAGSHUB_USERNAME
dvc remote modify dagshub --local password $env:DAGSHUB_TOKEN

# 4. Track your data files with DVC
# This creates .dvc pointer files that git DOES track
# while the actual Parquet files go to DagsHub
dvc add data/raw/aqi_features_historical.parquet
dvc add data/raw/aqi_features_live.parquet
dvc add data/interim/aqi_features_cleaned.parquet

# 5. Push data to DagsHub remote
Write-Host "Pushing data to DagsHub..." -ForegroundColor Cyan
dvc push

# 6. Commit the .dvc pointer files to git
git add data/raw/aqi_features_historical.parquet.dvc
git add data/raw/aqi_features_live.parquet.dvc
git add data/interim/aqi_features_cleaned.parquet.dvc
git add .dvc/config
git add .gitignore
git add dvc.yaml
git add .github/

git commit -m "feat: add DVC data versioning + GitHub Actions CI/CD"
git push origin main

Write-Host ""
Write-Host "DVC setup complete!" -ForegroundColor Green
Write-Host "Your data is now versioned on DagsHub."
Write-Host "GitHub Actions will use DVC to pull/push data on every run."
