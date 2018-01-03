# for Python 3
# Get Stock Data

# Based on the documentation at:
# https://pandas-datareader.readthedocs.io/en/latest/index.html

import datetime
import pandas_datareader as pdr
import requests_cache

expire_after = datetime.timedelta(days=30)
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)

# housekeeping
session.cache.remove_old_entries(datetime.datetime.utcnow() - expire_after)

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2013, 1, 27)

f = pdr.data.DataReader("F", 'yahoo', start, end, session=session)

print(f.ix['2010-01-04'])
