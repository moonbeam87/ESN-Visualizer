import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import wikipedia
from plotly.subplots import make_subplots
from datetime import datetime

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(
                data=[go.Candlestick(x=df['Date'],
                open=df['AAPL.Open'],
                high=df['AAPL.High'],
                low=df['AAPL.Low'],
                close=df['AAPL.Close'])])

external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']
app = dash.Dash("Stonks", external_stylesheets=external_stylesheets)
app.title = 'Stonks'



app.layout = html.Div(children=[
    html.H1(children='Stonks'),
    dcc.Graph(
        id='my-graph-1',
        figure = fig,
        
    ),
])
#Run App
if __name__ == '__main__':
    app.run_server(debug=True)