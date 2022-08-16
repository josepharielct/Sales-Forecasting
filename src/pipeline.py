import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
def mape(y_true:pd.Series,y_pred:pd.Series)->float:
    """
    Parameters:
    y_true    :    Pandas Series, Numpy Array
        Actual values of a variabel. i.e. Sales

    y_pred    :    Pandas Series, Numpy Array
        Predicted values of a variabel. i.e. Sales

    Returns:
    MAPE Value    :    float
        Mean Absolute Percentage Value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/ y_true)) *100
    
def append_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
    df1    :    Pandas DataFrame
        dataframe training
    
    df2    :    Pandas DataFrame
        dataframe testing

    Returns:
    df3    :    Pandas DataFrame, csv
        dataframe untuk forecast
    """
    df3 = df1.append(df2, ignore_index=True)
    return df3

def wrangle(path: str)->pd.DataFrame:
    """
    Parameters:
    path    :    str, path object or file-like object
        path untuk sebuah file csv

    Returns:
    df_tmp    :    Pandas DataFrame
        dataframe yang telah diformatkan sesuai dengan kaidah penamaan prophet
    """
    df_tmp = pd.read_csv(path)
    df_tmp['order_date'] =  pd.to_datetime(df_tmp['order_date'],format = '%d/%m/%Y')
    df_tmp.sort_values(by='order_date')
    df_tmp.reset_index(drop=True)
    df_tmp.rename(columns = {'order_date':'ds', 'sales_total':'y'}, inplace = True)
    return df_tmp

def add_aov(df:pd.DataFrame):
    """
    Parameters:
    df    :    Pandas DataFrame
        dataframe yang ingin ditambahkan feature AOV

    Returns:
    df['AOV']    :    Pandas Column
        Column AOV
    """
    df['AOV'] = df['y']/df['total_order']
    
def aov_range(x:float)->str:
    """
    Parameters:
    x    :    float
        AOV value

    Returns:
    'low', 'medium', 'medium-high', 'high'    :    string
        The category that AOV value belongs to
    """
    #Dibagi 4 Quartile, Q3 dibuang
    if x <= 6.806514e+05:
        return 'low'
    elif 6.806514e+05 < x <= 7.538420e+05:
        return 'medium'
    elif 7.538420e+05< x <= 8.184672e+05:
        return 'medium-high'
    elif x > 8.184672e+05:
        return 'high'
    
def nambah_aov(df:pd.DataFrame)->pd.DataFrame:
    """
    Parameters:
    df    :    Pandas DataFrame
        Dataframe yang ingin dibuat column aov kategorik

    Returns:
    df    :    Pandas DataFrame
        Dataframe yang telah ditambahkan column aov kategorik
    """
    df['aov_kategorik'] = df['AOV'].apply(lambda x: aov_range(x))
    return df
def encode(df:pd.DataFrame)->pd.DataFrame:
    """
    Parameters:
    df    :    Pandas DataFrame
        DataFrame yang memiliki column aov_kategorik

    Returns:
    df    :    Pandas DataFrame
        DataFrame yang telah ditambahkan aov_kategorik yang telah melalui one hot encoder
    """
    df["aov_kategorik"] = df["aov_kategorik"].astype('category')
    df = pd.get_dummies(df,columns=['aov_kategorik'])
    return df


def model_train_test(df1:pd.DataFrame, df2:pd.DataFrame,df3:pd.DataFrame)->json:
    """
    Parameters:
    df1    :    Pandas DataFrame
        DataFrame Training
    
    df2    :    Pandas DataFrame
        DataFrame Testing
    
    df3    :    Pandas DataFrame
        DataFrame Forecasting

    Returns:
    metrics    :    json
        json containing error values.
    """
    mod = Prophet()
    mod.add_regressor('aov_kategorik_high')
    mod.add_regressor('aov_kategorik_medium')
    mod.add_regressor('aov_kategorik_low')
    model_prophet = mod.fit(df1)
    fut = mod.make_future_dataframe(periods = 7,freq='D')
    fut['aov_kategorik_high'] = df3['aov_kategorik_high']
    fut['aov_kategorik_medium'] = df3['aov_kategorik_medium']
    fut['aov_kategorik_low'] = df3['aov_kategorik_low']
    forecast = mod.predict(fut)
    predictions = forecast['yhat'].iloc[-7:]
    rmse_result = mean_squared_error(df2.y, predictions, squared = False)
    mae_result = mean_absolute_error(df2.y, predictions)
    mape_result = mape(df2.y, predictions)
    diff = sum(df2.y) - sum(predictions)
    percent_diff = abs(sum(df2.y) - sum(predictions))/sum(df2.y) *100
    metrics = {"RMSE": rmse_result,"MAE": mae_result,"MAPE": mape_result, 'Actual - Predictions': diff, 'Percent Difference': percent_diff}
    # print('Performance Metrics:')
    # print('RMSE: ',rmse_result)
    # print('MAE:  ', mae_result)
    # print('MAPE: ', mape_result)
    # print('-----------------------')
    # print('Difference in Testing (Aggregate Weekly)')
    # print(f'Aktual - Prediksi: {diff}')
    # print(f'Percent Difference: {percent_diff}')
    with open('metric.json', 'w') as fp:
        json.dump(metrics, fp)
def add_aov_kategorik(df:pd.DataFrame)-> pd.Series:
    """
    Parameters:
    df    :    Pandas DataFrame
        DataFrame Forecasting
        
    Returns:
    aov_low    :    Pandas Series; Numpy Array; List, csv
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori low dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting
    
    aov_medium    :    Pandas Series; Numpy Array; List, csv
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori medium dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting
    
    aov_med_high    :    Pandas Series; Numpy Array; List, csv
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori medium-high dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting ****
    
    aov_high    :    Pandas Series; Numpy Array; List, csv
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori highdari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting
    """
    num_days = 7
    zeros = pd.Series(np.zeros(num_days))
    s_low = df['aov_kategorik_low']
    s_medium = df['aov_kategorik_medium']
    s_medium_high = df['aov_kategorik_medium-high']
    s_high = df['aov_kategorik_high']
    aov_low = pd.concat([s_low, zeros],ignore_index=True)
    aov_medium = pd.concat([s_medium, zeros],ignore_index=True)
    aov_med_high = pd.concat([s_medium_high, zeros],ignore_index=True)
    aov_high = pd.concat([s_high, zeros],ignore_index=True)
    return aov_low,aov_medium,aov_med_high,aov_high 
def process_df_aov_forecast(df:pd.DataFrame)->list:
    """
    Parameters:
    df    :    Pandas DataFrame
        DataFrame Forecast yang memiliki value AOV

    Returns:
    low    :    list
        List berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori low dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting

    medium    :    list
        List berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori medium dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting

    med_high    :    list
        List berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori medium-high dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting

    high    :    list
        List berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori high dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting
    """
    df_def = df[['ds','AOV']]
    df_def = df_def.rename(columns = {'AOV': 'y'})
    mod = Prophet()
    mod.fit(df_def)
    fut_aov = mod.make_future_dataframe(periods = 7, freq = 'D')
    forecast_aov = mod.predict(fut_aov)
    forecast_aov = pd.DataFrame(forecast_aov['yhat'].iloc[-7:])
    forecast_aov = forecast_aov.rename(columns = {'yhat':'AOV'})
    forecast_aov = nambah_aov(forecast_aov)
    forecast_aov = encode(forecast_aov)
    if 'aov_kategorik_low' not in list(forecast_aov.columns):
        forecast_aov['aov_kategorik_low'] = np.zeros(7, dtype=int)
    if 'aov_kategorik_medium' not in list(forecast_aov.columns):
        forecast_aov['aov_kategorik_medium'] = np.zeros(7, dtype=int)
    if 'aov_kategorik_medium-high' not in list(forecast_aov.columns):
        forecast_aov['aov_kategorik_medium-high'] = np.zeros(7, dtype=int)
    if 'aov_kategorik_high' not in list(forecast_aov.columns):
        forecast_aov['aov_kategorik_high'] = np.zeros(7, dtype=int)
    low = df['aov_kategorik_low'].tolist() + forecast_aov['aov_kategorik_low'].tolist()
    medium = df['aov_kategorik_medium'].tolist() + forecast_aov['aov_kategorik_medium'].values.tolist()
    med_high = df['aov_kategorik_medium-high'].tolist() + forecast_aov['aov_kategorik_medium-high'].values.tolist()
    high = df['aov_kategorik_high'].tolist() + forecast_aov['aov_kategorik_high'].values.tolist()
    return low,medium,med_high,high
def model_forecast_default(df:pd.DataFrame,aov_high:list,aov_medium:list,aov_low:list)->pd.DataFrame:
    """
    Parameters:
    df    :    Pandas DataFrame
        DataFrame untuk forecasting

    aov_high    :    Pandas Series; Numpy Array; List
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori high dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting

    aov_medium    :    Pandas Series; Numpy Array; List
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori medium dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting

    aov_low    :    Pandas Series; Numpy Array; List
        Pandas series berisi Binary Value (0,1) yang menandakan apakah AOV pada hari tersebut termasuk kategori low dari awal timeframe DataFrame hingga 7 hari sehabis tanggal terakhir di dataframe forecasting
    
    Returns:
    forecast    :    Pandas DataFrame, csv
        Pandas DataFrame yang berisi tanggal dan sales hasil forecast
    """
    num_days = 7
    m = Prophet()
    m.add_regressor('aov_kategorik_high')
    m.add_regressor('aov_kategorik_medium')
    m.add_regressor('aov_kategorik_low')
    model = m.fit(df)
    fut = m.make_future_dataframe(periods=num_days,freq='D')
    fut['aov_kategorik_high'] = aov_high
    fut['aov_kategorik_medium'] = aov_medium
    fut['aov_kategorik_low'] = aov_low
    forecast = m.predict(fut)
    forecast = forecast[['ds','yhat']]
    return forecast
def main():
    #Wrangle & Mengubah Format DataFrame
    df1 = wrangle('Dataset Train Mei.csv')
    df2 = wrangle('mei.csv')
    df3 = append_df(df1,df2)
    #Menambahkan Fitur AOV Numerik
    add_aov(df1)
    add_aov(df2)
    add_aov(df3)
    #Menambahkan Fitur AOV Kategorik
    df1 = nambah_aov(df1)
    df2 = nambah_aov(df2)
    df3 = nambah_aov(df3)
    df1 = encode(df1)
    df2 = encode(df2)
    df3 = encode(df3)
    #Memprediksi AOV 7 hari kedepan menggunakan Prophet
    l,m,mh,h = process_df_aov_forecast(df3)
    #Forecast Sales 7 hari kedepan menggunakan AOV yang diprediksi
    forecast_default = model_forecast_default(df3,l,m,h)
    forecast_default.to_csv(r'forecast_default.csv',index=False,header=True)
    
    #Training & Testing Untuk Mendapatkan Error Rates
    model_train_test(df1,df2,df3)

    #Membuat array untuk store aov untuk dash app
    aov_low,aov_med,aov_med_high,aov_high = add_aov_kategorik(df3)
    
    df3.to_csv(r'df_forecast_1.csv', index = False, header=True)
    aov_low.to_csv(r'aov_low.csv',index = False, header=True)
    aov_med.to_csv(r'aov_med.csv', index = False, header=True)
    aov_med_high.to_csv(r'aov_med_high.csv', index = False, header = True)
    aov_high.to_csv(r'aov_high.csv', index = False, header = True)


main()