# for Python 3
# Make Training Data

import datetime
import pandas as pd


# Raw data
Location = r'C:\Data\code\anntt\code_tutorial\HistoricalQuotes.csv'
df = pd.read_csv(Location)

#print(df.ix['2018-01-04'])
#print(df.head(10))

# Day of the week
#print(df['date'] )
# maybe a neural network would be easier to train with the days of the week expanded
dow_expanded = {0: [1,0,0,0,0], 1:[0,1,0,0,0], 2:[0,0,1,0,0], 3:[0,0,0,1,0], 4:[0,0,0,0,1]}

df['dow_code'] = pd.to_datetime(df['date'].values).dayofweek
#df['dow'] = df['dow_code'].apply(lambda x: dow_expanded[x])

#print(df.dtypes)
#print( f.columns)

print(df.head(10))

# short EMA (Exponential Moving Average)
ema_short = df['close'].ewm(span=9, adjust=False).mean()

print('short ema')
print(ema_short.head(10))

# long EMA (Exponential Moving Average)
ema_long = df['close'].ewm(span=200, adjust=False).mean()
print('long ema')
print(ema_long.head(10))