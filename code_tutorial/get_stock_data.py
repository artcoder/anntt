# for Python 3
# Get Stock Data

# Based on the documentation at:
# https://pandas-datareader.readthedocs.io/en/latest/index.html

import pandas as pd
import pandas_datareader as pdr
import datetime
import requests_cache
import matplotlib.pyplot as plt

expire_after = datetime.timedelta(days=7)
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2017, 1, 27)

prices = pdr.data.DataReader('AAPL', 'yahoo', start, end, session=session)

print (prices.tail())

ema_150 = prices['Adj Close'].ewm(span=150, adjust=False, min_periods=150).mean()
ema_9 = prices['Adj Close'].ewm(span=9, adjust=False, min_periods=9).mean()

plt.title('Get Stock Data')
prices['Adj Close'].plot(grid=True, label='AAPL Adj close')
ema_150.plot(grid=True, label='150 day EMA')
ema_9.plot(grid=True, label='9 day EMA')
plt.legend()
plt.draw()

# Make a blocking call to keep the pyplot window open
plt.show()