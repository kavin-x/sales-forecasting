from flask import Flask
from flask import Flask,jsonify,request
from prophet import Prophet
from pandas import read_csv
from matplotlib import pyplot
from pandas import to_datetime
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error
import json

app = Flask(__name__)

@app.route('/')
def predict_sales():
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
    df = read_csv(path, header=0)
    # prepare expected column names
    df.columns = ['ds', 'y']
    df['ds']= to_datetime(df['ds'])
    # define the model
    model = Prophet()
    # fit the model
    model.fit(df)
    # define the period for which we want a prediction
    future = list()
    for i in range(1, 13):
        date = '1969-%02d' % i
        future.append([date])
    print(type(future))
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
    mae = mean_absolute_error(y_true, y_pred)
    print(y_pred,y_true)
    print('MAE: %.3f' % mae)
    # plot expected vs actual
    pyplot.plot(y_true, label='Actual')
    pyplot.plot(y_pred, label='Predicted')
    pyplot.legend()
    pyplot.show()
    return json.dumps(mae)

if __name__ == '__main__':
    app.run()