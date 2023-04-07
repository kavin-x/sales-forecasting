from flask import Flask
from prophet import Prophet
from pandas import read_csv
from matplotlib import pyplot
from pandas import to_datetime
from pandas import DataFrame
app = Flask(__name__)

@app.route('/')
def hello_world():
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
    future = DataFrame(future)
    future.columns = ['ds']
    future['ds']= to_datetime(future['ds'])
    # use the model to make a forecast
    forecast = model.predict(future)
    # summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    model.plot(forecast)
    pyplot.show()
    return 'Hello World!'

@app.route('/app')
def hello_app():
    return 'Hello app!'

if __name__ == '__main__':
    app.run()