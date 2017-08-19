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
from datetime import datetime,timedelta

from pandas.tslib import Timestamp
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

def average(series):
    return float(sum(series))/len(series)

def average(series, n=None):
    if n is None:
        return average(series, len(series))
    return float(sum(series[-n:]))/n

def moving_average(series, n):
    return average(series[-n:])

def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    #ser =  pd.Series(result,index = series.index.tolist())
    return result

def double_exponential_smoothing(series, alpha, beta):
    time_list = series.index.tolist()
    print len(time_list)
    length = len(time_list)
    to_date = str(ts.index.tolist()[length-1])
    from_date = str(ts.index.tolist()[length-2])

    datetime_object1 = datetime.strptime(from_date, '%Y-%m-%d %H:%M:%S')
    datetime_object2 = datetime.strptime(to_date, '%Y-%m-%d %H:%M:%S')
    diff = (datetime_object2-datetime_object1).total_seconds()
    s_len = 10
    for n in range(0,s_len):
        datetime_object3 = datetime.strptime(str(time_list[len(time_list)-1]), '%Y-%m-%d %H:%M:%S')
        time_list.append(Timestamp(datetime_object3 + timedelta(seconds=int(diff))))


    print len(time_list)
    result = [series[0]]
    for n in range(1, len(series)+s_len):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    ser =  pd.Series(result,index = time_list)
    return ser


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    
    time_list = series.index.tolist()
    print len(time_list)
    length = len(time_list)
    to_date = str(ts.index.tolist()[length-1])
    from_date = str(ts.index.tolist()[length-2])

    datetime_object1 = datetime.strptime(from_date, '%Y-%m-%d %H:%M:%S')
    datetime_object2 = datetime.strptime(to_date, '%Y-%m-%d %H:%M:%S')
    diff = (datetime_object2-datetime_object1).total_seconds()
    for n in range(0,n_preds):
        datetime_object3 = datetime.strptime(str(time_list[len(time_list)-1]), '%Y-%m-%d %H:%M:%S')
        time_list.append(Timestamp(datetime_object3 + timedelta(seconds=int(diff))))


    print len(time_list)
    
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    ser =  pd.Series(result,index = time_list)
    return ser


length = len(ts)
weights = [0.1, 0.2, 0.3, 0.4]


print type(exponential_smoothing(ts, 0.1))
print type(exponential_smoothing(ts, 0.9))

predict1 = exponential_smoothing(ts, 0.5)

predict2 = exponential_smoothing(ts, 0.9)

predict3 = double_exponential_smoothing(ts, alpha=0.3, beta=0.3)

predict4 = triple_exponential_smoothing(ts, 24, 0.3, 0.029, 0.3, 24)

"""
plt.subplot(411)
plt.plot(predict1, label='predict1')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(predict2, label='predict2')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(ts,label='ts')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(predict3,label='Predict3')
plt.legend(loc='best')
"""
plt.plot(ts)
plt.plot(predict4, color='red')

#print len(ts)
#print len(predict1)
#print len(predict2)
#print len(predict3)
#print len(predict4)





