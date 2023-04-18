from flask import Flask, make_response, request
from io import StringIO
import csv
from prophet import Prophet
from pandas import read_csv,date_range
from matplotlib import pyplot
from pandas import to_datetime
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from flask_cors import CORS
from flask import request
import requests
import pandas as pd
from datetime import datetime, timedelta
app = Flask(__name__)

CORS(app)# This will enable CORS for all routes

@app.route('/',methods=['GET', 'POST'])
def predict_sales():

    currentYear,month,date=int(datetime.today().strftime('%Y')),int(datetime.today().strftime('%m')),int(datetime.today().strftime('%d'))
    start_date = datetime(currentYear, month, date)
    period = request.files['period']
    # Getting List of weeks using pandas
    if period=="week":
        end_date = datetime(currentYear+2, 1, 10)
        month_list = pd.period_range(start=start_date, end=end_date, freq='w')
        month_list = [month.strftime("%Y-%m-%d") for month in month_list]
    # Getting List of Months using pandas
    if period=="month":
        end_date = datetime(currentYear+2, 1, 10)
        month_list = pd.period_range(start=start_date, end=end_date, freq='m')
        month_list = [month.strftime("%Y-%m-%d") for month in month_list]
    
    # define the period for which we want a prediction
    future = list()
    for i in month_list:
        future.append([i])
    print(future)
    csv_file = request.files['file']
    # req = requests.get('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv')
    # url_content=req.content
    # csv_file=open("downloaded.csv","wb")
    # csv_file.write(url_content)

    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
    df = read_csv(csv_file.stream)
    # prepare expected column names
    df.columns = ['ds', 'y']
    df['ds']= to_datetime(df['ds'])
    # define the model
    model = Prophet()
    # fit the model
    model.fit(df)

    print(future)
    future = DataFrame(future)
    future.columns = ['ds']
    future['ds']= to_datetime(future['ds'])
    # use the model to make a forecast
    forecast = model.predict(future)
    d = forecast.to_dict(orient='records')
    # summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    model.plot(forecast)
    pyplot.show()
    forecast['ds'] = forecast['ds'].astype(str)
    return json.dumps(forecast.to_dict(orient='records'))
    

@app.route('/app')
def mean_absolute_error():
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
    df = read_csv(path, header=0)
    # prepare expected column names
    df.columns = ['ds', 'y']
    df['ds']= to_datetime(df['ds'])
    # create test dataset, remove last 12 months
    train = df.drop(df.index[-12:])
    print(train.tail())
    # define the model
    model = Prophet()
    # fit the model
    model.fit(train)
    # define the period for which we want a prediction
    future = list()
    for i in range(1, 13):
        date = '1968-%02d' % i
        future.append([date])
    future = DataFrame(future)
    future.columns = ['ds']
    future['ds'] = to_datetime(future['ds'])
    # use the model to make a forecast
    forecast = model.predict(future)
    # calculate MAE between expected and predicted values for december
    y_true = df['y'][-12:].values
    y_pred = forecast['yhat'].values
    mae = mean_squared_error(y_true, y_pred)
    print('MAE: %.3f' % mae)
    # plot expected vs actual
    pyplot.plot(y_true, label='Actual')
    pyplot.plot(y_pred, label='Predicted')
    pyplot.legend()
    pyplot.show()
    return json.dumps(mae)

if __name__ == '__main__':
    app.run()