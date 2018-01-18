# for Python 3
# Make Training Data

import datetime
import pandas_datareader as pdr
import pandas as pd
import requests_cache

expire_after = datetime.timedelta(days=30)
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)

# housekeeping
#session.cache.remove_old_entries(datetime.datetime.utcnow() - expire_after)

dow_expanded = {0: [1,0,0,0,0], 1:[0,1,0,0,0], 2:[0,0,1,0,0], 3:[0,0,0,1,0],
4:[0,0,0,0,1]}

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2013, 1, 27)

f = pdr.data.DataReader("F", 'yahoo', start, end, session=session)

#print(f.ix['2010-01-04'])

f['dow_code'] = pd.to_datetime(f.index.values).dayofweek
f['dow'] = f['dow_code'].apply(lambda x: dow_expanded[x])

print(f.dtypes)
#print( f.columns)

print(f.head(5))

