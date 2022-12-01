#
# Stock Prediction
#

## Import needed modules
import requests
import pandas as pd
import numpy  as np
import datetime as dt
import matplotlib.pyplot as plt 
import matplotlib.dates  as mdates
from os.path import exists

# LSTM libraries
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

# measuring performance
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

# dataset normalization and spliting
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split



## Download the netflix dataset if it doesn't exist
if not exists("./NFLX.csv") :
    print("[+] Downloading NETFLIX dataset.")
    resp = requests.get("https://query1.finance.yahoo.com/v7/finance/download/NFLX?period1=1506988800&period2=1664755200&interval=1d&events=history&includeAdjustedClose=true")
    open("NFLX.csv", 'wb').write(resp.content)

#  import dataset
print("[+] Importing the dataset")
dataset_stock = pd.read_csv('./NFLX.csv', index_col='Date')

print(("\n[*] Overview of the dataset."))
print(dataset_stock.head())

print(("\n[*] Description of the dataset."))
print(dataset_stock.describe())

#  Show a graph
print("\n[+] Plot a graph of the dataset.")
# Format the dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=180))
dates = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dataset_stock.index.values]
# add the high values to the graph
plt.plot(dates, dataset_stock['High'], label='High')
plt.plot(dates, dataset_stock['Low'], label='Low')
plt.xlabel('Time period')
plt.ylabel('Stock price')
plt.gcf().autofmt_xdate()
# show legend
plt.legend()
plt.show()


## Scaling, Spliting dataset for training and testing
target   = dataset_stock['Close']

data_scaler = MinMaxScaler(feature_range=(0,1))
x_features  = data_scaler.fit_transform(np.array(target).reshape(-1,1))


split_idx       = int(np.ceil(len(x_features) * 0.8))
testing_size    = len(target) - split_idx
train_d, test_d = x_features[0:split_idx:], x_features[split_idx: len(x_features): 1]

def split(data_, step=1):
    X, Y = [], []
    for idx in range(len(data_) - step - 1):
        X.append(data_[idx: (idx + step),0 ])
        Y.append(data_[idx + step , 0])
    return np.array(X), np.array(Y)

X_train, y_train    = split(train_d)
X_test, y_test      = split(train_d)
X_train             = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test              = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

## LSTM model training
lstm = Sequential()
lstm.add(LSTM(40, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')


lstm.fit(X_train, y_train, epochs=80, batch_size=5,verbose=1)

# predict
prediction = lstm.predict(X_test)

## accuracy
mape = mean_absolute_percentage_error(y_test, prediction)
mse  = mean_squared_error(y_test, prediction, squared=False)

print("The accuracy of the model.")
print(f'MAPE :: {mape}')
print(f'RMSE :: {mse}')