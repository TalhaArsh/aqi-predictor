"""
Feast Setup Script
==================
Run once after installing feast to initialize the feature store
and materialize historical features to the online store.

Usage:
    python setup_feast.py
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    # 1. Ensure data/feast directory exists
    Path("data/feast").mkdir(parents=True, exist_ok=True)
    logger.info("Created data/feast/")

    # 2. Run feast apply (registers feature definitions)
    logger.info("Running feast apply (registering feature definitions)...")
    result = subprocess.run(
        [sys.executable, "-m", "feast", "-c", "feature_store", "apply"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"feast apply failed:\n{result.stderr}")
        return
    logger.info(result.stdout)
    logger.info("✅ Feature definitions registered")

    # 3. Materialize offline → online store
    logger.info("Materializing offline features to online store...")
    from datetime import datetime, timezone, timedelta
    from src.feast_utils import materialize_offline_to_online

    # Push last 2 years to online store
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=730)
    materialize_offline_to_online(start_date=start, end_date=end)

    logger.info("\n✅ Feast setup complete!")
    logger.info("Feature store is ready at: feature_store/")
    logger.info("Online store at:           data/feast/online_store.db")
    logger.info("Registry at:               data/feast/registry.db")
    logger.info("\nNext steps:")
    logger.info("  python src/feature_pipeline.py   (writes to Feast each hour)")
    logger.info("  python -m src.training.data_loader (reads from Feast for training)")


if __name__ == "__main__":
    main()
