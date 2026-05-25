"""
Data Exploration Script
========================
Loads the historical Parquet and prints a comprehensive report:
- Shape, columns, dtypes
- Null counts per column
- AQI distribution + EPA category breakdown
- Per-column statistics (min/mean/median/std/max)
- Sample rows (head, tail, random)
- Time coverage analysis (gaps, duplicates)
- Target column sanity (deltas + absolutes)
- Pollutant cross-correlations with AQI

Run anytime after backfill to inspect what you actually have.

Usage:
  python src/explore_data.py                                   # historical
  python src/explore_data.py --file data/raw/aqi_features_live.parquet
  python src/explore_data.py --file data/interim/aqi_features_cleaned.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = "data/raw/aqi_features_historical.parquet"


def section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ----------------------------- Reports -----------------------------
def report_shape(df: pd.DataFrame, path: Path):
    section("1. DATASET OVERVIEW")
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"File:    {path}")
    print(f"Size:    {size_mb:.2f} MB")
    print(f"Shape:   {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB in RAM")


def report_columns(df: pd.DataFrame):
    section("2. COLUMNS + DTYPES")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    print(info.to_string())


def report_time_coverage(df: pd.DataFrame):
    section("3. TIME COVERAGE")
    if "timestamp" not in df.columns:
        print("  No 'timestamp' column.")
        return

    ts = pd.to_datetime(df["timestamp"])
    print(f"  Start:    {ts.min()}")
    print(f"  End:      {ts.max()}")
    print(f"  Span:     {(ts.max() - ts.min()).days} days "
          f"({(ts.max() - ts.min()).total_seconds() / 3600:.0f} hours)")
    print(f"  Rows:     {len(df):,}")

    expected_hours = int((ts.max() - ts.min()).total_seconds() / 3600) + 1
    coverage = len(df) / expected_hours * 100
    print(f"  Expected: {expected_hours:,} hourly rows")
    print(f"  Coverage: {coverage:.1f}%")

    # Duplicates
    dups = df.duplicated(subset="timestamp").sum()
    print(f"  Duplicates on timestamp: {dups}")

    # Gaps between consecutive timestamps
    sorted_ts = ts.sort_values().reset_index(drop=True)
    deltas = sorted_ts.diff().dt.total_seconds() / 3600
    gaps = deltas[deltas > 1]
    print(f"  Gaps > 1 hour: {len(gaps)}")
    if len(gaps) > 0:
        print(f"    Largest gap: {gaps.max():.1f} hours")
        print(f"    Total missing hours: {(gaps - 1).sum():.0f}")


def report_aqi(df: pd.DataFrame):
    section("4. AQI QUALITY + DISTRIBUTION")
    if "aqi" not in df.columns:
        print("  No 'aqi' column.")
        return

    aqi = df["aqi"]
    aqi_valid = aqi.dropna()
    print(f"  Valid rows: {len(aqi_valid):,} / {len(df):,} "
          f"({aqi_valid.notna().mean() * 100:.1f}%)")
    print(f"  Range:      {aqi_valid.min():.0f} → {aqi_valid.max():.0f}")
    print(f"  Mean:       {aqi_valid.mean():.1f}")
    print(f"  Median:     {aqi_valid.median():.1f}")
    print(f"  Std:        {aqi_valid.std():.1f}")
    print(f"  Skewness:   {aqi_valid.skew():.2f}  (positive=right-tailed)")

    print(f"\n  Threshold counts:")
    print(f"    AQI ≥ 100  (Unhealthy for sensitive): {(aqi_valid >= 100).sum():,} "
          f"({(aqi_valid >= 100).mean() * 100:.1f}%)")
    print(f"    AQI ≥ 150  (Unhealthy):                {(aqi_valid >= 150).sum():,} "
          f"({(aqi_valid >= 150).mean() * 100:.1f}%)")
    print(f"    AQI ≥ 200  (Very unhealthy):           {(aqi_valid >= 200).sum():,} "
          f"({(aqi_valid >= 200).mean() * 100:.1f}%)")
    print(f"    AQI ≥ 300  (Hazardous):                {(aqi_valid >= 300).sum():,} "
          f"({(aqi_valid >= 300).mean() * 100:.1f}%)")
    print(f"    AQI = 500  (Max):                      {(aqi_valid >= 500).sum():,}")

    print(f"\n  EPA category breakdown:")
    bins = [-1, 50, 100, 150, 200, 300, 501]
    labels = ["Good (0-50)", "Moderate (51-100)", "Unhealthy-Sensitive (101-150)",
              "Unhealthy (151-200)", "Very Unhealthy (201-300)", "Hazardous (301+)"]
    cats = pd.cut(aqi_valid, bins=bins, labels=labels)
    counts = cats.value_counts().reindex(labels, fill_value=0)
    pct = (counts / len(aqi_valid) * 100).round(2)
    for label, n, p in zip(labels, counts, pct):
        bar = "█" * int(p / 2)
        print(f"    {label:<32} {n:>6,} ({p:5.2f}%) {bar}")


def report_pollutants(df: pd.DataFrame):
    section("5. POLLUTANTS (μg/m³)")
    pollutants = ["pm25", "pm10", "no2", "o3", "co", "so2", "nh3"]
    rows = []
    for col in pollutants:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            rows.append({"pollutant": col, "n": 0})
            continue
        rows.append({
            "pollutant": col.upper(),
            "n_valid": len(s),
            "null%": round(df[col].isna().mean() * 100, 1),
            "min": round(s.min(), 1),
            "median": round(s.median(), 1),
            "mean": round(s.mean(), 1),
            "max": round(s.max(), 1),
            "std": round(s.std(), 1),
        })
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def report_weather(df: pd.DataFrame):
    section("6. WEATHER (if available)")
    weather_cols = ["temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds"]
    any_data = False
    for col in weather_cols:
        if col not in df.columns:
            continue
        n_valid = df[col].notna().sum()
        if n_valid == 0:
            print(f"  {col:<14} → 100% null")
        else:
            any_data = True
            s = df[col].dropna()
            print(f"  {col:<14} n={n_valid:,}, range=[{s.min():.1f}, {s.max():.1f}], "
                  f"mean={s.mean():.1f}")
    if not any_data:
        print("  (All weather columns are null — historical weather endpoint timed out)")


def report_targets(df: pd.DataFrame):
    section("7. TARGET COLUMNS (training labels)")
    horizons = [24, 48, 72]
    for h in horizons:
        abs_col = f"aqi_t_plus_{h}h"
        delta_col = f"aqi_delta_{h}h"
        if abs_col in df.columns:
            a = df[abs_col].dropna()
            print(f"\n  {abs_col}  (absolute future AQI)")
            print(f"    n={len(a):,}, mean={a.mean():.1f}, std={a.std():.1f}, "
                  f"range=[{a.min():.0f}, {a.max():.0f}]")
        if delta_col in df.columns:
            d = df[delta_col].dropna()
            print(f"  {delta_col}    (delta from current — used for training)")
            print(f"    n={len(d):,}, mean={d.mean():+.2f}, std={d.std():.1f}, "
                  f"range=[{d.min():+.0f}, {d.max():+.0f}]")
            print(f"    abs(delta) median: {d.abs().median():.1f}, "
                  f"95th pct: {d.abs().quantile(0.95):.1f}")


def report_correlations(df: pd.DataFrame):
    section("8. CORRELATIONS WITH AQI")
    if "aqi" not in df.columns:
        return
    numeric_cols = [c for c in df.columns
                    if df[c].dtype in ("float64", "int64") and c != "aqi"]
    corrs = df[["aqi"] + numeric_cols].corr()["aqi"].drop("aqi")
    corrs = corrs.dropna().sort_values(key=abs, ascending=False)
    print(f"  Top 15 features correlated with AQI:")
    for feat, val in corrs.head(15).items():
        direction = "+" if val > 0 else "-"
        bar = "█" * int(abs(val) * 30)
        print(f"    {feat:<25} {direction}{abs(val):.3f}  {bar}")


def report_samples(df: pd.DataFrame):
    section("9. SAMPLE ROWS")
    cols = [c for c in ["timestamp", "aqi", "pm25", "pm10",
                        "aqi_lag_1h", "aqi_rolling_24h",
                        "aqi_t_plus_24h", "aqi_delta_24h"]
            if c in df.columns]

    print("\n  FIRST 5 rows:")
    print(df[cols].head().to_string(index=False))

    print("\n  LAST 5 rows:")
    print(df[cols].tail().to_string(index=False))

    print("\n  RANDOM 5 rows:")
    print(df[cols].sample(min(5, len(df)), random_state=0).to_string(index=False))


def report_top_aqi_events(df: pd.DataFrame):
    section("10. WORST AQI HOURS")
    if "aqi" not in df.columns:
        return
    cols = [c for c in ["timestamp", "aqi", "pm25", "pm10", "hour", "month"]
            if c in df.columns]
    worst = df.nlargest(10, "aqi")[cols]
    print(worst.to_string(index=False))


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Explore an AQI Parquet file")
    parser.add_argument("--file", default=DEFAULT_PATH, help="Path to the parquet file")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"📂 Loading {path}...")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    report_shape(df, path)
    report_columns(df)
    report_time_coverage(df)
    report_aqi(df)
    report_pollutants(df)
    report_weather(df)
    report_targets(df)
    report_correlations(df)
    report_samples(df)
    report_top_aqi_events(df)

    print("\n" + "=" * 72)
    print("  ✅ Exploration complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
