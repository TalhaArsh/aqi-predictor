import os
import redis

host = 'innovative-microquiet-birthday-21764.db.redis.io'
port = 16572
password = os.environ.get('REDIS_PASSWORD', '')

# Load from .env if not in environment
if not password:
    try:
        with open('.env') as f:
            for line in f:
                if line.startswith('REDIS_PASSWORD='):
                    password = line.strip().split('=', 1)[1]
    except:
        pass

r = redis.Redis(host=host, port=port, password=password, decode_responses=True)
data = r.hgetall('aqi_predictor:features:Karachi')
print('Redis features found:', len(data))
print('AQI:', data.get('aqi'))
print('Temperature:', data.get('temperature'))
print('wind_speed:', data.get('wind_speed'))
print('aqi_lag_1h:', data.get('aqi_lag_1h', 'NOT YET (need 1h history)'))

