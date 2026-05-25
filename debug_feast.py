"""Debug script to check what's actually in the Feast online store."""
import sys
sys.path.insert(0, '.')

from feast import FeatureStore
import pandas as pd
import numpy as np

store = FeatureStore(repo_path="feature_store")

print("=== Feature Views ===")
for fv in store.list_feature_views():
    print(f"  {fv.name}: entities={[e for e in fv.entities]}")

print("\n=== Trying different entity key formats ===")
test_entities = [
    {"city": "Karachi"},
    {"city": b"Karachi"},  # bytes format
]

features_to_fetch = [
    "aqi_features:aqi",
    "aqi_features:pm25",
    "weather_features:temperature",
]

for entity_row in test_entities:
    print(f"\nTrying entity: {entity_row}")
    try:
        result = store.get_online_features(
            features=features_to_fetch,
            entity_rows=[entity_row],
        ).to_dict()
        print(f"  Result keys: {list(result.keys())}")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== Checking SQLite directly ===")
import sqlite3
from pathlib import Path

db_path = Path("data/feast/online_store.db")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    for table in tables:
        tname = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM '{tname}'")
        count = cursor.fetchone()[0]
        print(f"  {tname}: {count} rows")
        if count > 0:
            cursor.execute(f"SELECT * FROM '{tname}' LIMIT 1")
            row = cursor.fetchone()
            print(f"  Sample row keys: {[desc[0] for desc in cursor.description]}")
    conn.close()
else:
    print(f"Database not found at {db_path}")
