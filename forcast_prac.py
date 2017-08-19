import pprint
import json
import collections
import json
from random import randint
import threading
import time

from influxdb import DataFrameClient
import json
import dateutil.parser as parser
import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
#%matplotlib inline
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


rcParams['figure.figsize'] = 15, 6

host='influxdb1'
port=8086
user = ''
password = ''
dbname = 'telegraf'

client = DataFrameClient(host, port, user, password, dbname)
query="SELECT mean(usage_user) FROM cpu  WHERE cpu = 'cpu-total' AND host='cephmonnode1.devops.allcolo.com' AND time > now() - 7d GROUP BY time(1h) fill(0)"
data = client.query(query)
dataframe = data['cpu']
list = dataframe.index.tolist()
list_final = []
for x in list:
	date = (parser.parse(str(x)))
	iso = date.isoformat()
	inter_date = iso.split("+")
	dt = datetime.strptime(inter_date[0], "%Y-%m-%dT%H:%M:%S")
	dt64 = np.datetime64(dt)
	list_final.append(dt64)

dataframe2 = pd.DataFrame({'mean' : dataframe['mean'].tolist()},index=list_final)
ts = dataframe2['mean'] 

print ts.head(10)

#plt.plot(ts)


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=24)
    rolstd = pd.rolling_std(timeseries, window=24)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

#test_stationarity(ts)

#ts_log = np.log(ts)
#plt.plot(ts_log)

#moving_avg = pd.rolling_mean(ts_log,12)
#plt.plot(ts_log)
#plt.plot(moving_avg, color='red')

#ts_log_moving_avg_diff = ts_log - moving_avg
#print ts_log_moving_avg_diff.head(12)

#ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_moving_avg_diff)


#expwighted_avg = pd.ewma(ts_log, halflife=12)
#plt.plot(ts_log)
#plt.plot(expwighted_avg, color='red')

#ts_log_ewma_diff = ts_log - expwighted_avg
#test_stationarity(ts_log_ewma_diff)

#ts_log_diff = ts_log - ts_log.shift()
#plt.plot(ts_log_diff)

#ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)

ts_diff = ts - ts.shift()
#plt.plot(ts_diff)

ts_diff.dropna(inplace=True)
#test_stationarity(ts_diff)
"""
decomposition = seasonal_decompose(ts)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
"""
"""
lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
"""

model = ARIMA(ts, order=(1, 1, 1))  
results_ARIMA = model.fit(disp=-1)
print(results_ARIMA.summary())

length = len(ts)

s = length
e = length+10

predicted_values = results_ARIMA.predict(start=s,end=e)

values = results_ARIMA.fittedvalues.append(predicted_values)

predictions_ARIMA_diff = pd.Series(values, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA = pd.Series(data=ts.ix[0], index=values.index)


predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0)

plt.plot(ts)
plt.plot(predictions_ARIMA)



