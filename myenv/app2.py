import dash
from dash import dcc, html
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import plotly.graph_objs as go
import joblib
import psutil
import requests
import time

# Initialize Dash app
app = dash.Dash(__name__)

# Load the data
df = pd.read_csv('system_usage_data_30_days_2.csv')
df.columns = df.columns.str.strip()

# Layout of the app
app.layout = html.Div(style={'backgroundColor': '#000000', 'padding': '20px'}, children=[
    html.H1("System Metrics Dashboard", style={'textAlign': 'center', 'color': 'white'}),
    html.Div([
        # CPU Usage Bar Graph
        dcc.Graph(
            id='cpu-usage-bar',
            figure={
                'data': [
                    go.Bar(
                        x=df['Time-Date'],
                        y=df['CPU Usage (%)'],
                        name='CPU Usage (%)',
                        marker_color='red'
                    )
                ],
                'layout': go.Layout(
                    title="CPU Usage Over Time",
                    xaxis={'title': 'Time-Date', 'color': 'white'},
                    yaxis={'title': 'CPU Usage (%)', 'color': 'white'},
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(color='white')
                )
            }
        ),
        # Memory Usage Bar Graph
        dcc.Graph(
            id='memory-usage-bar',
            figure={
                'data': [
                    go.Bar(
                        x=df['Time-Date'],
                        y=df['Memory Usage (%)'],
                        name='Memory Usage (%)',
                        marker_color='blue'
                    )
                ],
                'layout': go.Layout(
                    title="Memory Usage Over Time",
                    xaxis={'title': 'Time-Date', 'color': 'white'},
                    yaxis={'title': 'Memory Usage (%)', 'color': 'white'},
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(color='white')
                )
            }
        ),
    ], style={'width': '90%', 'margin': 'auto'})
])

# FastAPI app for prediction
api_app = FastAPI()
model = joblib.load('auto_scaler_model.pkl')

class UsageData(BaseModel):
    cpu_usage: float
    memory_usage: float
    user_count: int

api_app.mount("/dashboard", WSGIMiddleware(app.server))

@api_app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}

@api_app.post("/predict/")
async def predict(data: UsageData):
    features = [[data.cpu_usage, data.memory_usage, data.user_count]]
    prediction = model.predict(features)
    return {
        "message": "Prediction completed!",
        "received_data": data.dict(),
        "predicted_scaling": prediction.tolist()
    }

# Monitor system metrics and send to the API
url = "http://13.232.212.55:8000/predict/"

CPU_THRESHOLD = 5   # in percentage
MEMORY_THRESHOLD = 50

while True:
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        user_count = len(psutil.users())

        if cpu_usage > CPU_THRESHOLD or memory_usage > MEMORY_THRESHOLD:
            data = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "user_count": user_count
            }

            print(f"Sending Data to API: {data}")
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print(f"Prediction Received: {response.json()}")
            else:
                print(f"Failed to get a valid response. Status Code: {response.status_code}")
        else:
            print(f"Conditions not met. CPU: {cpu_usage}%, Memory: {memory_usage}%")
    except requests.exceptions.RequestException as e:
        print(f"Error while sending request: {e}")
    time.sleep(300)

