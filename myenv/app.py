import dash
from dash import dcc, html
import pandas as pd
from fastapi.middleware.wsgi import WSGIMiddleware
import plotly.graph_objs as go
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize Dash app
app = dash.Dash(__name__)

# Load the data
df = pd.read_csv('system_usage_data_30_days_2.csv')
print(df.columns.tolist())
df.columns = df.columns.str.strip()

# Layout of the app
app.layout = html.Div(style={'backgroundColor': '#000000', 'padding': '20px'}, children=[
    html.H1(
        "System Metrics Dashboard",
        style={'textAlign': 'center', 'color': 'white'}
    ),

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
    ], style={'width': '90%', 'margin': 'auto'}),
])
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
        # Extract the features from the request data
    features = [[data.cpu_usage, data.memory_usage, data.user_count]]
                
                    # Get prediction using the trained model
    prediction = model.predict(features)  # Assuming your model has a `predict` method
                            
                                # Return the prediction and received dat
    return {
        "message": "Prediction completed!",
        "received_data": data.dict(),
        "predicted_scaling": prediction.tolist()
        }
