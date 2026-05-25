"""
AQI Predictor - Phase 3.6: Select & Promote Best Model Per Horizon
====================================================================
Reads the local metrics CSVs (ridge_metrics.csv, rf_metrics.csv, lstm_metrics.csv),
picks the model with the lowest test_RMSE for each horizon, and promotes that
version in MLflow's Model Registry to the "Production" stage.

After this runs, the dashboard can simply load:
  - "ridge_24h@Production", "rf_24h@Production", "lstm_multihorizon@Production"
without needing to know which version is best.
"""

import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient
import dagshub

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = "aqi-predictor"
MODELS_DIR = Path("models")

METRICS_FILES = {
    "ridge": MODELS_DIR / "ridge_metrics.csv",
    "rf":    MODELS_DIR / "rf_metrics.csv",
    "lstm":  MODELS_DIR / "lstm_metrics.csv",
}


def init_dagshub():
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)


def load_all_metrics() -> pd.DataFrame:
    frames = []
    for fam, path in METRICS_FILES.items():
        if not path.exists():
            logger.warning(f"{path} missing — skipping {fam}.")
            continue
        df = pd.read_csv(path)
        df["family"] = fam
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No metrics CSVs found — train models first.")
    return pd.concat(frames, ignore_index=True)


def best_per_horizon(metrics: pd.DataFrame) -> pd.DataFrame:
    """Lowest test_RMSE wins per horizon."""
    return (metrics.sort_values("test_RMSE")
                   .groupby("horizon_h", as_index=False)
                   .first())


def promote_to_production(client: MlflowClient, model_name: str):
    """Promote the latest version of `model_name` to Production stage,
    archiving any previous Production version."""
    try:
        latest = client.get_latest_versions(model_name, stages=["None", "Staging"])
        if not latest:
            # Already in Production — get all and pick the highest version
            latest = client.search_model_versions(f"name='{model_name}'")
            if not latest:
                logger.warning(f"  No versions found for {model_name}")
                return
        # Pick the highest version number
        latest_version = max(latest, key=lambda mv: int(mv.version))

        # Archive existing Production versions
        for mv in client.get_latest_versions(model_name, stages=["Production"]):
            if mv.version != latest_version.version:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived"
                )
                logger.info(f"  Archived {model_name} v{mv.version}")

        # Promote
        client.transition_model_version_stage(
            name=model_name, version=latest_version.version, stage="Production"
        )
        logger.info(f"  ✅ Promoted {model_name} v{latest_version.version} → Production")
    except Exception as e:
        logger.warning(f"  Could not promote {model_name}: {e}")


def main():
    init_dagshub()
    client = MlflowClient()

    metrics = load_all_metrics()
    best = best_per_horizon(metrics)

    print("\n" + "=" * 72)
    print("BEST MODEL PER HORIZON (selected by lowest test_RMSE)")
    print("=" * 72)
    print(best[["horizon_h", "family", "model", "test_RMSE", "test_MAE", "test_R2"]]
          .to_string(index=False))

    print("\n" + "=" * 72)
    print("PROMOTING WINNERS TO PRODUCTION")
    print("=" * 72)

    for _, row in best.iterrows():
        h = int(row["horizon_h"])
        family = row["family"]
        if family == "lstm":
            # LSTM is a single multi-output model
            model_name = "lstm_multihorizon"
        else:
            model_name = f"{family}_{h}h"
        print(f"\nHorizon {h}h → winner: {family} (RMSE={row['test_RMSE']:.2f})")
        promote_to_production(client, model_name)

    print(f"\n✅ DagsHub registry: "
          f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/models")


if __name__ == "__main__":
    main()
