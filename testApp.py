import datetime as dt
import os
import time
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from flask_caching import Cache
import warnings
warnings.filterwarnings('ignore')
from pyESN import ESN 

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']

app = dash.Dash("ESN Visualizer", external_stylesheets=external_stylesheets)
app.title = 'ESN Visualizer'

theme = {
    'dark': False,
    'detail': '#007439',
    'primary': '#00EA64', 
    'secondary': '#6E6E6E'
}

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60

@cache.memoize(timeout=TIMEOUT)
def get_dataframe(name):
    df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + name +'&apikey=WCXVE7BAD668SJHL&datatype=csv')
    clone = df
    df = df.rename(columns={"timestamp":"Date"})
    df = df.set_index(df['Date'])
    df = df.sort_index()
    df = df.drop(columns=['open', 'low', 'high', 'volume', 'Date'])
    return df


def get_series():
    names = ['AAPL', 'GOOGL', 'FB', 'IBM', 'AMZN']
    series = []
    for name in names:
        df = get_dataframe(name)
        
        series.append(df)
    stocks = pd.concat(series, axis = 1)
    stocks.columns = ['AAPL', 'GOOGL','FB','IBM', 'AMZN']
    stocks['Date'] = stocks.index

    return stocks

def calculate_ESN(name, rand_seed, nReservoir, spectralRadius, future, futureTotal):
    data = open(name+".txt").read().split()
    data = np.array(data).astype('float64')
    sparsity=0.2
    noise = .0005
    nReservoir = nReservoir *1
    spectralRadius = spectralRadius * 1
    future = future * 1
    futureTotal = futureTotal * 1



    esn = ESN(n_inputs = 1,
        n_outputs = 1, 
        n_reservoir = nReservoir,
        sparsity=sparsity,
        random_state=rand_seed,
        spectral_radius = spectralRadius,
        noise=noise)

    trainlen = data.__len__()-futureTotal
    pred_tot=np.zeros(futureTotal)

    for i in range(0,futureTotal,future):
        pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
        prediction = esn.predict(np.ones(future))
        pred_tot[i:i+future] = prediction[:,0]
    return pred_tot

app.layout = html.Div(id='dark-theme-components', children=[

    html.H1(children='ESN Visualizer'),
    html.H4(children='Please give the graph time to load, ESN Calculations may take while'),
    daq.ToggleSwitch(
        id='graph-color',
        label='This Toggle is Useless - Just like You! :)',
        style={'width': '250px', 'margin': 'auto'}, 
        value=True
    ),

    dcc.Input(
            id="rand-seed", type="number", value=23,
            debounce=True, placeholder="Random Seed",
    ),
    html.H1(children=' '),
    daq.NumericInput(
        id='n-resevoir',
        value=500,
        label="N Resevoir",
    ),
    html.H1(children=' '),
    daq.NumericInput(
        id='spectral-radius',
        value=1.2,
        label="Spectral Radius",
    ),
    html.H1(children=' '),
    daq.NumericInput(
        id='future',
        value=120,
        label="Future",
    ),
    html.H1(children=' '),
    daq.NumericInput(
        id='future-total',
        value=120,
        label="Future Total",
    ),
    html.H1(children=''),
    dcc.Dropdown(
        id='live-dropdown',
        value='AAPL',
        multi=False,
        options=[{'label': i, 'value': i} for i in get_series().columns]
    ),
    dcc.Graph(id='live-graph'),
    dcc.Graph(id='ESN-graph')
], style={'padding': '50px'})

@app.callback(Output('live-graph', 'figure'),
              [Input('live-dropdown', 'value'),
              Input('graph-color', 'color')]
)
def update_live_graph(value, color):
    df = get_series()
    return {
        'data': [{
            'x': df['Date'],
            'y': df[value],
            'line': {
                'width': 1,
                'color': '#FF0000',
                'shape': 'spline'
            }
        }],
        'layout': {
                'title': 'Stock Data'
            }
    }

@app.callback(Output('ESN-graph', 'figure'),
              [Input('live-dropdown', 'value'),
              Input('rand-seed', 'random_seed'),
              Input('n-resevoir', 'n_reservoir'),
              Input('spectral-radius', 'spectral_radius'),
              Input('future', 'future_value'),
              Input('future-total', 'future_total'),]
)
def update_ESN_graph(value, random_seed, n_reservoir, spectral_radius, future_value, future_total):
    df = get_series()
    ESNData = calculate_ESN(value, random_seed, n_reservoir, spectral_radius, future_value, future_total)
    return {
        'data': [{
            'x': df['Date'],
            'y': ESNData,
            'line': {
                'width': 1,
                'color': '#0000FF',
                'shape': 'spline'
            }
        }],
        'layout': {
                'title': 'ESN Predicted Data'
            }
    }



if __name__ == '__main__':
    app.run_server(debug=True)
