"""
Quick check: what's currently in Redis Cloud?
Run this anytime to verify the feature pipeline is working.

Usage: python check_redis.py
"""
import os
import redis
from datetime import datetime

# Load from .env if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

host     = os.getenv("REDIS_HOST", "innovative-microquiet-birthday-21764.db.redis.io")
port     = int(os.getenv("REDIS_PORT", "16572"))
password = os.getenv("REDIS_PASSWORD", "")

print(f"Connecting to Redis: {host}:{port}")
r = redis.Redis(host=host, port=port, password=password, decode_responses=True)

try:
    r.ping()
    print("✅ Connected to Redis Cloud\n")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

key = "aqi_predictor:features:Karachi"
data = r.hgetall(key)

if not data:
    print("❌ No data found in Redis")
    print(f"   Key checked: {key}")
    exit(1)

print(f"✅ Found {len(data)} features in Redis")
print(f"   Key: {key}")

# Show TTL
ttl = r.ttl(key)
print(f"   TTL: {ttl}s ({ttl//60} min until expiry)\n")

# Key metrics
print("── Current conditions ──────────────────")
for feat in ["aqi", "pm25", "pm10", "temperature", "humidity",
             "wind_speed", "pressure", "cloud_cover"]:
    val = data.get(feat, "NOT FOUND")
    print(f"  {feat:<20} {val}")

print("\n── AQI lag features ────────────────────")
for feat in ["aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h",
             "aqi_lag_24h", "aqi_rolling_3h", "aqi_change_1h"]:
    val = data.get(feat, "NOT YET (need more history)")
    print(f"  {feat:<20} {val}")

print("\n── Weather lag features ─────────────────")
for feat in ["temperature_lag_1h", "temperature_lag_6h",
             "wind_speed_lag_1h", "humidity_lag_6h"]:
    val = data.get(feat, "NOT YET")
    print(f"  {feat:<20} {val}")

# Timestamp
ts = data.get("_timestamp")
if ts:
    dt = datetime.fromtimestamp(float(ts))
    print(f"\n── Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S')} local time")

print(f"\n── Feature count breakdown ─────────────")
populated = sum(1 for v in data.values() if v and v != "nan")
print(f"  Populated features: {populated}/96")
print(f"  Missing (NaN lags): {96 - populated}")
print(f"\n  These will auto-populate as history builds:")
print(f"  • After 1h:  aqi_lag_1h, weather_lag_1h")
print(f"  • After 6h:  lag_6h, rolling_6h")
print(f"  • After 24h: lag_24h, rolling_24h")
print(f"  • After 72h: lag_72h, rolling_72h (full 96 features)")
