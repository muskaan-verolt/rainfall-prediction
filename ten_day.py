import json
import pickle
import requests
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sklearn
import os
import pickle
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#import tensorflow
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Bidirectional



from datetime import datetime, timedelta
import pandas as pd

def forecast(location: str, date: str,  feature: str):
    # Convert string date to datetime object
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    print(date_obj)
    # Set time to 9am and 3pm for the preprocessed date
    date_9am = date_obj.replace(hour=9, minute=0, second=0, microsecond=0)
    date_3pm = date_obj.replace(hour=15, minute=0, second=0, microsecond=0)

    # Create DataFrame for next ten days with time set to 9am
    date_range_9am = [date_9am + timedelta(days=i) for i in range(10)]
    next_ten_days_9am = pd.DataFrame({'Datetime': date_range_9am})
    next_ten_days_9am['Date'] = next_ten_days_9am['Datetime'].dt.date

    # Create DataFrame for next ten days with time set to 3pm
    date_range_3pm = [date_3pm + timedelta(days=i) for i in range(10)]
    next_ten_days_3pm = pd.DataFrame({'Datetime': date_range_3pm})
    next_ten_days_3pm['Date'] = next_ten_days_3pm['Datetime'].dt.date

    # Merge the two DataFrames based on the date column
    next_ten_days = pd.merge(next_ten_days_9am, next_ten_days_3pm, on='Datetime', suffixes=('_9am', '_3pm'))
    next_ten_days = pd.concat([next_ten_days_9am, next_ten_days_3pm], ignore_index=True)

    # Sort the values based on the datetime column
    next_ten_days = next_ten_days.sort_values(by='Datetime').reset_index(drop=True)

    # Preprocess the data for prediction
    next_ten_days['DayOfWeek'] = next_ten_days['Datetime'].dt.dayofweek
    next_ten_days['Day'] = next_ten_days['Datetime'].dt.day
    next_ten_days['Month'] = next_ten_days['Datetime'].dt.month
    next_ten_days['Year'] = next_ten_days['Datetime'].dt.year

    if feature not in ['precip', 'weather']:
        next_ten_days['HourOfDay'] = next_ten_days['Datetime'].dt.hour

    
    # Define the directory path where the models are stored
    directory = f"{feature}_models//"
    
    # Construct the file path for the pickle file
    file_path = os.path.join(directory, f"{location}.pkl")
    
    try:
        # Attempt to open and load the pickle file
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            
    except FileNotFoundError:
        print(f"No model found for location '{location}' and feature '{feature}'.")
        return None
    
    if feature not in ['precip', 'weather']:
        X_pred = np.array(next_ten_days[['DayOfWeek', 'Day', 'Month', 'Year', 'HourOfDay']])
    else:
        X_pred = np.array(next_ten_days[['DayOfWeek', 'Day', 'Month', 'Year']])
    
    ## Reshape input features for LSTM
    X_pred = np.reshape(X_pred, (X_pred.shape[0], 1, X_pred.shape[1]))

    feature_scaled = model.predict(X_pred)
    
    # Inverse transform the scaled temperatures
    
    with open(f'{feature}_scaler.z', 'rb') as f:
        feature_scaler = load(f)
    feature_preds = feature_scaler.inverse_transform(feature_scaled)

    if feature in ('weather', 'wd'):
        with open(f'label_encoder_{feature}.z', 'rb') as f:
            le_encoder = load(f)
        og = []
        for pred in feature_preds:
            og.append(le_encoder.classes_[int(np.round(pred))])
        return og
    
    
    return  feature_preds.flatten()


def temp_forecast(location: str, date: str):
    # Convert string date to datetime object
    
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    print(date_obj)
    date_loop = date_obj
    next_ten_days=pd.DataFrame(columns=['DayOfWeek','Day', 'Month', 'Year', 'HourOfDay'])
    for date_obj in [date_loop + timedelta(days=i) for i in range(10)]:
        # Set time to 9am and 3pm for the preprocessed date
        date_5am = date_obj.replace(hour=5, minute=0, second=0, microsecond=0)
        date_9am = date_obj.replace(hour=9, minute=0, second=0, microsecond=0)
        date_12am = date_obj.replace(hour=12, minute=0, second=0, microsecond=0)
        date_3pm = date_obj.replace(hour=15, minute=0, second=0, microsecond=0)
        date_4pm = date_obj.replace(hour=14, minute=0, second=0, microsecond=0)
        

        
        for i in [date_5am, date_9am, date_12am, date_3pm, date_4pm]:

            # Preprocess the data for prediction
            dw =i.weekday()
            day = i.day
            mon = i.month
            yr = i.year
            hod = i.hour

            next_ten_days.loc[len(next_ten_days.index)] = [dw, day, mon, yr, hod]

    print(next_ten_days)
    # Define the directory path where the models are stored
    directory = "tf_models//"
    
    # Construct the file path for the pickle file
    file_path = os.path.join(directory, f"{location}.pkl")
    
    try:
        # Attempt to open and load the pickle file
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            
    except FileNotFoundError:
        print(f"No model found for location '{location}' and tf.")
        return None
    
    X_pred = np.array(next_ten_days[['DayOfWeek', 'Day', 'Month', 'Year', 'HourOfDay']])
    print(X_pred)
    
    ## Reshape input features for LSTM
    X_pred = np.reshape(X_pred, (X_pred.shape[0], 1, X_pred.shape[1]))

    feature_scaled = model.predict(X_pred)
    
    # Inverse transform the scaled temperatures
    
    with open('tf_scaler.z', 'rb') as f:
        feature_scaler = load(f)
    feature_preds = feature_scaler.inverse_transform(feature_scaled)
    
    return  feature_preds.flatten()


#tf_forecasts = temp_forecast('Sydney', '2024-05-16')
#print(tf_forecasts)


'''
location = 'Albury'
date='2024-05-13'
temp_forecasts = forecast(location, date, 'temp')
cloud_forecasts = forecast(location, date, 'cloud')
pressure_forecasts = forecast(location, date, 'pressure')
humidity_forecasts = forecast(location, date, 'humidity')
ws_forecasts = forecast(location, date, 'ws')
dew_forecasts = forecast(location, date, 'dew')
wd_forecasts = forecast(location, date, 'wd')
weather_forecasts = forecast(location, date, 'weather')
print(temp_forecasts)
print(cloud_forecasts)
print(humidity_forecasts)
print(pressure_forecasts)
print(ws_forecasts)
print(dew_forecasts)
print(wd_forecasts)
print(weather_forecasts)

'''
