from fastapi import FastAPI, Request, Form
from fastapi.params import Body
from fastapi.responses import HTMLResponse, RedirectResponse
import json
import input_features
import pickle
import sklearn
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import warnings
import ten_day
import today
from typing import List, Dict
import requests

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=UserWarning)


app = FastAPI()

app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

class WeatherDataRequest(BaseModel):
    location: str
    date: str

class TodayResponse(BaseModel):
    date: List[str]
    day_precip: List[float]
    precipitation_amount: List[float]
    conditions: List[str]
    temp_min: List[float]
    temp_max: List[float]
    sunrise_time: List[str]
    sunset_time: List[str]
    humidity: List[float]
    wind_speed: List[float]
    wind_direction: List[float]
    dew_point: List[float]
    pressure: List[float]
    icons: List[str]
    latitude: float
    longitude: float

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: str
    

class ForecastResponse(BaseModel):
    forecasts: List[List[float]]
    cards: List[List[str]]

async def get_weather_data_async():
    # Call the function from today.py to extract weather data
    weather_data = await today.today_data()
    return weather_data




@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_weather_data")
async def get_weather_data(request_data: WeatherDataRequest):
    location = request_data.location
    date = request_data.date
    print(f"Location: {type(location)}")
    print(f"Date: {type(date)}")
    # Fetch weather data and process features
    response_data = input_features.weather_data(location, date)
    print("Response Taken!")
    return response_data

@app.post("/rainfall_prediction")
async def rainfall_prediction(request_data: WeatherDataRequest):
    response_data = await get_weather_data(request_data)
    features = input_features.get_input_features(response_data)
    print(f"Response Data: {type(response_data)}")
    print(f"Features: {type(features)}")
    # Load model
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction, probabs = input_features.is_it_gonna_rain_tomorrow(features, model)
    print(f"Prediction {type(prediction)}")
    print(f"Probabs: {type(probabs)}")
    if prediction == 0:
        prediction = "No"
        probabs = str(round(float(probabs[1]) * 100, 2))
    else:
        prediction="Yes"
        probabs= str(round(float(probabs[1]) * 100, 2))
    
    print(f"Prediction {type(prediction)}")
    respons= PredictionResponse(prediction=prediction, probabilities=probabs)
    return respons


@app.post("/get_forecasts")
async def ten_day_forecast(req: WeatherDataRequest):
    location = req.location
    date = req.date
    temp_forecasts = ten_day.temp_forecast(location, date)
    cloud_forecasts = ten_day.forecast(location, date, 'cloud')
    pressure_forecasts = ten_day.forecast(location, date, 'pressure')
    humidity_forecasts = ten_day.forecast(location, date, 'humidity')
    ws_forecasts = ten_day.forecast(location, date, 'ws')
    dew_forecasts = ten_day.forecast(location, date, 'dew')
    wd_forecasts = ten_day.forecast(location, date, 'wd')
    todays = today.today_data()
    weather_forecasts = todays['conditions']
    icons_forecasts = todays['icons']
    precipitation_amount = todays['precipitation_amount']
    dates = todays['date']
    print(temp_forecasts)
    print(cloud_forecasts)
    print(humidity_forecasts)
    print(pressure_forecasts)
    print(ws_forecasts)
    print(dew_forecasts)
    print(weather_forecasts)
    cards=[weather_forecasts, icons_forecasts, dates, wd_forecasts]
    print(cards)
    forecasts = [temp_forecasts, (cloud_forecasts/8)*100, humidity_forecasts, pressure_forecasts, ws_forecasts, dew_forecasts, precipitation_amount]
    return ForecastResponse(forecasts=forecasts, cards=cards)

@app.post("/today")
async def get_today_data(req: WeatherDataRequest):
    response_data = await get_weather_data(req)
    weather_data = today.today_data()  
    return TodayResponse(**weather_data)


#pred, prob = get_weather_data(WeatherDataRequest(location=loc, date=dat))
#if pred == 0:
#    print("No")
#    print("Probability:", prob[0]*100, "%")
#else:
#    print("Yes")
#    print("Probability:", prob[1]*100, "%")



#url = f"https://api.openweathermap.org/data/2.5/forecast?lat=39.099724&lon=-94.578331&appid={API_KEY}"
#url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat=39.099724&lon=-94.578331&dt=2020-03-04&appid={API_KEY}"
##url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Albury/2008-12-02/2008-12-03?unitGroup=us&include=alerts%2Cdays%2Chours%2Ccurrent&key=GBZSSX3SKWTV6UWW2D2525KY3&contentType=json"
#url = f"https://api.openweathermap.org/data/2.5/weather?lat=39.099724&lon=-94.578331&appid={API_KEY}"
#response = requests.get(url)
#print(response.json())
#@app.post("/createposts")
#def create_post(payload: dict = Body(...)):
#    print(payload)
#    return {"message":f" title: {payload['title']}, content : {payload['content']}" }'''