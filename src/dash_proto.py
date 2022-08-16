from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from prophet import Prophet
import json
app = Dash(__name__)

df = pd.read_csv('df_forecast_1.csv')
forecast_default = pd.read_csv('forecast_default.csv')
forecast_default['ds'] = pd.to_datetime(forecast_default['ds'],format = '%Y-%m-%d')
aov_low = pd.read_csv('aov_low.csv')
aov_med = pd.read_csv('aov_med.csv')
aov_med_high = pd.read_csv('aov_med_high.csv')
aov_high = pd.read_csv('aov_high.csv')
with open('metric.json') as json_file:
    metrics = json.load(json_file)
app.layout = html.Div([
    dcc.RadioItems(
        ['1 (AOV < 680.651)', '2 (680.651 < AOV < 753.842)', '3 (753.842 < AOV < 818.467,2)', '4 (AOV >  818.467,2)'],
        '1 (AOV < 680.651)',
        id='aov_category',
        inline=True
    ), 
    html.P(([key for key in metrics.keys()][0],' : ', "{:,}".format(int(round([value for value in metrics.values()][0],0))),' | Akar Kuadrat rata-rata jarak antara hasil prediksi dan aktual - Lebih sensitif terhadap impulses/spikes dibandingkan kedua metrik lainnya, angka RMSE yang tinggi menunjukkan bahwa data aktual banyak yang berbeda dengan data training')),
    html.P(([key for key in metrics.keys()][1],' : ', "{:,}".format(int(round([value for value in metrics.values()][1],0))), ' | Rata-rata jarak antara hasil prediksi dan data aktual   | Average trend secara general')),
    html.P(([key for key in metrics.keys()][2],' : ', round([value for value in metrics.values()][2],2),'%','        | MAE yang diubah menjadi bentuk persen')),
    html.P(([key for key in metrics.keys()][3],' : ', 'Rp ',"{:,}".format(int(round([value for value in metrics.values()][3],0))))),
    html.P(([key for key in metrics.keys()][4],' : ', round([value for value in metrics.values()][4],2), '%')),
    dcc.Graph(id='indicator-graphic')
])
@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('aov_category', 'value')
)
def update_graph(aov_category:str):
    """
    Parameters:
    aov_category    :    string
        Kategori aov yang dipilih user berdasarkan radio button

    Returns:
    fig    :    figure
        update figure pada dash app
    """
    pilih_kategori = int(aov_category[0])
    low =list(np.concatenate(aov_low[:-7].values.tolist()))
    med = list(np.concatenate(aov_med[:-7].values.tolist()))
    med_high = list(np.concatenate(aov_med_high[:-7].values.tolist()))
    high =list(np.concatenate(aov_high[:-7].values.tolist()))
    ones = np.ones(7).tolist()
    zeroes = np.zeros(7).tolist()
    if pilih_kategori == 1:
        low = low + zeroes
        med = med + ones
        med_high = med_high + zeroes
        high = high + zeroes
    elif pilih_kategori == 2:
        low = low + zeroes
        med = med + zeroes
        med_high = med_high + ones
        high = high + zeroes
    elif pilih_kategori == 3:
        low = low + ones
        med = med + zeroes
        med_high = med_high + zeroes
        high = high + zeroes
    elif pilih_kategori == 4:
        low = low + zeroes
        med = med + zeroes
        med_high = med_high + zeroes
        high = high + ones
    num_days = 7
    m = Prophet()
    m.add_regressor('aov_kategorik_high')
    m.add_regressor('aov_kategorik_medium')
    m.add_regressor('aov_kategorik_low')
    m.fit(df)
    fut = m.make_future_dataframe(periods=num_days,freq='D')
    fut['aov_kategorik_high'] = high
    fut['aov_kategorik_medium'] = med
    fut['aov_kategorik_low'] = low
    forecast = m.predict(fut)
    forecast_next_week = forecast[['ds','yhat_lower','yhat_upper','yhat']].iloc[-num_days:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"].iloc[365:], y=forecast["yhat"].iloc[365:], name="Forecast", mode="lines"))
    fig.add_trace(go.Scatter(x=df["ds"].iloc[365:], y=df["y"].iloc[365:], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast_default["ds"].iloc[365:], y=forecast_default["yhat"].iloc[365:], name="Default Settings", mode="lines"))
    fig.update_layout(
        title="Sales Forecast", xaxis_title="Date", yaxis_title="Sales"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)