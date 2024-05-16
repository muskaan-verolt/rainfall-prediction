import json
import pickle
import requests
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sklearn


pd.set_option('display.max_columns', None)
with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
with open('encoded_values.pkl', 'rb') as f:
    encoded_values_dict = pickle.load(f)


def get_api_key():
    with open('api_key.json', 'r') as f:
        config = json.load(f)
        return config['API_KEYS']['main_api']

API_KEY = get_api_key()

def weather_data(location: str, date: str):
    if API_KEY:
        date_dt = pd.to_datetime(date)
        date_end = (date_dt + timedelta(days=10)).strftime("%Y-%m-%d")
        
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date}/{date_end}?unitGroup=metric&include=alerts%2Cdays%2Chours%2Ccurrent&key={API_KEY}&contentType=json"
        response = requests.get(url)
        if response.status_code == 200:
            data_str = json.dumps(response.json())
            with open('api_response.json', 'w') as f:
                f.write(data_str)
            return response.json()
        else:
            with open('api_response.json', 'r') as f:
                response = f.read()
            return json.loads(response)
            
'''def weather_data(location: str, date: str):
    with open('api_response.json', 'r') as f:
        response = f.read()

    weather_data = json.loads(response)
    print('Response Taken from input features')
    print(weather_data)
    return weather_data'''
         


def get_input_features(response_data):
    print(response_data)
    weather_data = {
        'Date': response_data['days'][0]['datetime'],
        'Location': response_data['address'],
        'MinTemp': response_data['days'][0]['tempmin'],
        'MaxTemp': response_data['days'][0]['tempmax'],
        'Rainfall': response_data['days'][0]['precip'],
        'Sunrise': response_data['days'][0]['sunrise'],
        'Sunset': response_data['days'][0]['sunset'],
        'WindGustSpeed': response_data['days'][0]['windgust'],
        'WindSpeed9am': response_data['days'][0]['hours'][9]['windspeed'],
        'WindSpeed3pm': response_data['days'][0]['hours'][15]['windspeed'],
        'Humidity9am': response_data['days'][0]['hours'][9]['humidity'],
        'Humidity3pm': response_data['days'][0]['hours'][15]['humidity'],
        'Pressure9am': response_data['days'][0]['hours'][9]['pressure'],
        'Pressure3pm': response_data['days'][0]['hours'][15]['pressure'],
        'Cloud9am': response_data['days'][0]['hours'][9]['cloudcover'],
        'Cloud3pm': response_data['days'][0]['hours'][15]['cloudcover'],
        'Temp9am': response_data['days'][0]['hours'][9]['temp'],
        'Temp3pm': response_data['days'][0]['hours'][15]['temp'],
        'WindGustDir': response_data['days'][0]['winddir'],
        'WindDir9am': response_data['days'][0]['hours'][9]['winddir'],
        'WindDir3pm': response_data['days'][0]['hours'][15]['winddir'],
        #'RainToday': response_data['days'][0]['precip'],
        'dew9am': response_data['days'][0]['hours'][9]['dew'],
        'dew3pm': response_data['days'][0]['hours'][15]['dew'],
        }
    return weather_data

def is_it_gonna_rain_tomorrow(features, model):
    
    features_df = pd.DataFrame(features, index=[0])

    # Preprocess user input
    rainfall = features_df.copy()
    print(rainfall)
    
    # Convert direction degrees to cardinal
    def degrees_to_cardinal(d):
        cardinals = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round((d  + 11.25)/22.5 - 0.02) % 16
        return cardinals[index]

    rainfall['WindGustDir'] = rainfall['WindGustDir'].apply(degrees_to_cardinal)
    rainfall['WindDir9am'] = rainfall['WindDir9am'].apply(degrees_to_cardinal)
    rainfall['WindDir3pm'] = rainfall['WindDir3pm'].apply(degrees_to_cardinal)

    # Convert percentage of cloud cover to oktas
    rainfall['Cloud9am'] = rainfall['Cloud9am'] / 12.5
    rainfall['Cloud3pm'] = rainfall['Cloud3pm'] / 12.5

    # Create columns
    rainfall['Sunrise'] = pd.to_datetime(rainfall['Sunrise'], format='%H:%M:%S')
    rainfall['Sunset'] = pd.to_datetime(rainfall['Sunset'], format='%H:%M:%S')
    rainfall['Sunshine'] = (rainfall['Sunset'] - rainfall['Sunrise']).dt.total_seconds() / 3600

    rainfall['RainToday'] = rainfall['Rainfall'].apply(lambda x: 'Yes' if x > 1 else 'No')

    rainfall['WindSpeed9am']= rainfall['WindSpeed9am'].apply(lambda x: np.random.rand() * np.random.randint(25,30) if x > 30 else x) 
    #rainfall['Evaporation'] = rainfall['Evaporation'].apply(lambda x: np.random.rand() * np.random.randint(15,20) if x > 15 else x) 
    rainfall['Humidity9am']= rainfall['Humidity9am'].apply(lambda x: np.random.uniform(35, 55) if x < 35 else x) 
    rainfall['WindGustSpeed']= rainfall['WindGustSpeed'].apply(lambda x: np.random.uniform(70,80) if x > 80 else x) 
    rainfall['WindSpeed3pm']= rainfall['WindSpeed3pm'].apply(lambda x: np.random.uniform(40,48) if x > 48 else x) 
    
    rainfall['dew9am'] = (rainfall['dew9am'] * 9/5) + 32
    rainfall['dew3pm'] = (rainfall['dew3pm'] * 9/5) + 32
    
   
    rainfall['Date'] = pd.to_datetime(rainfall['Date'])
    rainfall['day'] = rainfall['Date'].dt.day
    rainfall['month'] = rainfall['Date'].dt.month
    rainfall['year'] = rainfall['Date'].dt.year 
    
    seasons = {1: 'Summer', 2: 'Summer', 3: 'Autumn', 4: 'Autumn', 5:'Autumn', 6:'Winter', 7: 'Winter', 8:'Winter', 9:'Spring', 10:'Spring', 11:'Spring', 12:'Summer'}
    rainfall['Season'] = rainfall['month'].map(seasons) 
    
    rainfall['TempRange'] = rainfall['MaxTemp'] - rainfall['MinTemp'] 
    rainfall['TempAvg'] = (rainfall['MaxTemp'] + rainfall['MinTemp']) /2 
    rainfall['WindSpeed12pm'] = (rainfall['WindSpeed9am'] + rainfall['WindSpeed3pm']) / 2
    rainfall['Humidity12pm'] = (rainfall['Humidity9am'] + rainfall['Humidity3pm']) / 2
    rainfall['Pressure12pm'] = (rainfall['Pressure9am'] + rainfall['Pressure3pm']) / 2
    rainfall['Cloud12pm'] = (rainfall['Cloud9am'] + rainfall['Cloud3pm']) / 2
    
    rainfall['Rained?'] = rainfall['Rainfall'].apply(lambda x: 'No' if x == 0 else 'Yes')
    
    def classify_weather(row):
        if row['Rainfall'] > 2.5:
            return 'Rainy' # 11000
        elif row['Cloud12pm'] > 6:
            return 'Cloudy' # 9800
        elif row['WindGustSpeed'] > 45 or row['WindSpeed12pm'] > 20:
            return 'Stormy' # 4800
        elif row['TempAvg'] > 25 and row['Humidity12pm'] < 50:
            return 'Clear' # 500
        else:
            return 'Partly Cloudy' # 5100
    
    rainfall['Weather'] = rainfall.apply(classify_weather, axis=1) 
    
    
    def bin_rainfall(row):
        if row['Rainfall'] > -0.1 and row['Rainfall'] < 0.001:
            return 'No Rain' 
        elif row['Rainfall'] > 0.001 and row['Rainfall'] < 2.5:
            return 'Slight Drizzle'
        elif row['Rainfall'] > 2.5 and row['Rainfall'] < 4:
            return 'Moderate Shower' 
        elif row['Rainfall'] > 4 and row['Rainfall'] < 10:
            return 'Heavy Rain' 
        else:
            return 'Violent Downpour' # 5100
    rainfall['rainfall_bins'] = rainfall.apply(bin_rainfall, axis=1)  

    cols_to_encode = [ 'WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday', 'Season', 'Rained?', 'Weather', 'rainfall_bins', 'Location']
    for feature in cols_to_encode:
        print(rainfall[feature])
        if feature in encoded_values_dict:
            encoded_values = encoded_values_dict[feature]
            rainfall[f'{feature}_n'] = round(rainfall[feature].map(encoded_values))
        else:
            print(f"Feature '{feature}' is not encoded. Please encode it before using.")

    rainfall['TH'] = rainfall['TempRange'] / rainfall['Humidity12pm']
    
    X = rainfall[['MinTemp', 'MaxTemp', 'Rainfall','Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'day', 'month', 'year', 'TempRange', 'TempAvg', 'WindSpeed12pm', 'Humidity12pm', 'Pressure12pm', 'Cloud12pm', 'TH', 'dew9am', 'dew3pm', 'WindGustDir_n', 'WindDir9am_n', 'WindDir3pm_n', 'RainToday_n', 'Season_n', 'Rained?_n', 'Weather_n', 'rainfall_bins_n', 'Location_n']]
    prediction = model.predict(X)
    predicted_probabilities = model.predict_proba(X)
    
    probabilities_dict = {}
    for idx, class_name in enumerate(model.classes_):
        probabilities_dict[class_name] = predicted_probabilities[:, idx][0]
    print(probabilities_dict)

    return prediction, probabilities_dict
   


