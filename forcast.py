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
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

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

ts_log = np.log(ts)
#plt.plot(ts_log)

moving_avg = pd.rolling_mean(ts_log,12)
#plt.plot(ts_log)
#plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
print ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_moving_avg_diff)


expwighted_avg = pd.ewma(ts_log, halflife=12)
#plt.plot(ts_log)
#plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
#test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
#plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)

decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
"""
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
#plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
#plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
#plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
"""
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
#test_stationarity(ts_log_decompose)


lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

print lag_acf
print lag_pacf

#Plot ACF: 
#plt.subplot(121) 
#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')

#Plot PACF:
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')
#plt.tight_layout()
"""
model = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
print results_AR
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
"""
"""
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

"""


model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1) 
print(results_ARIMA.summary())
#print type(results_ARIMA)
#print results_ARIMA.forecast()


length = len(ts_log)

s = length
e = length+10

#print results_ARIMA.fittedvalues

#print("--------------------------------------------")

predicted_values = results_ARIMA.predict(start=s,end=e)

values = results_ARIMA.fittedvalues.append(predicted_values)

#print values

#plt.plot(ts_log_diff)
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

#"""

predictions_ARIMA_diff = pd.Series(values, copy=True)
#print predictions_ARIMA_diff.head()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(data=ts_log.ix[0], index=values.index)

print ts_log.ix[0]
print("****************************************")
print ts_log

#print predictions_ARIMA_log
#print("***********************************************")
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print predictions_ARIMA_log
#print predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
#print("***********************************************************")
#print predictions_ARIMA
plt.plot(ts)
plt.plot(predictions_ARIMA)

li = (predictions_ARIMA-ts)
li.dropna(inplace=True)

s = sum(li**2)

plt.title('RMSE: %.4f'% np.sqrt(s/len(predictions_ARIMA)))
val = predictions_ARIMA.tolist()

ind = predictions_ARIMA.index.tolist()
ind2 = ts.index.tolist()

print len(ind)
print len(ind2)
print type(results_ARIMA)

print predictions_ARIMA.index.tolist()[0]
print predictions_ARIMA.index.tolist()[168]

print ts.index.tolist()[0]
print ts.index.tolist()[168]

# %run forcast.py




