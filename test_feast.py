from src.feast_utils import get_online_features
f = get_online_features('Karachi')
if f:
    print('Feast online store working')
    print('AQI:', f.get('aqi'))
    print('Temperature:', f.get('temperature'))
    print('Total features:', len(f))
else:
    print('No features returned - online store may be empty')

