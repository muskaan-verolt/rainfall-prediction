import json
from typing import List, Dict

def today_data()  -> Dict[str, Dict[str, float]]:
    with open('api_response.json', 'r') as f:
        api_response = f.read()
    
    weather_data = json.loads(api_response)
    
    parameter_data = {
        'date': [day['datetime'] for day in weather_data['days']],
        'day_precip': [hour['precip'] for hour in weather_data['days'][0]['hours']],
        'precipitation_amount': [sum(hour['precip'] for hour in day['hours']) for day in weather_data['days']],
        'conditions': [day['conditions'] for day in weather_data['days']],
        'temp_min': [day['tempmin'] for day in weather_data['days']],
        'temp_max': [day['tempmax'] for day in weather_data['days']],
        'sunrise_time': [day['sunrise'] for day in weather_data['days']],
        'sunset_time': [day['sunset'] for day in weather_data['days']],
        'humidity': [day['humidity'] for day in weather_data['days']],
        'wind_speed': [day['windspeed'] for day in weather_data['days']],
        'wind_direction': [day['winddir'] for day in weather_data['days']],
        'dew_point': [day['dew'] for day in weather_data['days']],
        'pressure': [day['pressure'] for day in weather_data['days']],
        'icons': [day['icon'] for day in weather_data['days']],
        'latitude': weather_data['latitude'],
        'longitude': weather_data['longitude']
    }

    return parameter_data

#print(today_data())