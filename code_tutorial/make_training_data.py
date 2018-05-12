# for Python 3
# Make Training Data

import datetime
import pandas as pd

# maybe a neural network would be easier to train with the days of the week expanded
dow_expanded = {0: [1,0,0,0,0], 1:[0,1,0,0,0], 2:[0,0,1,0,0], 3:[0,0,0,1,0], 4:[0,0,0,0,1]}

Location = r'C:\Data\code\anntt\code_tutorial\HistoricalQuotes.csv'
df = pd.read_csv(Location)

#print(df.ix['2018-01-04'])
#print(df.head(10))

#print(df['date'] )

df['dow_code'] = pd.to_datetime(df['date'].values).dayofweek
#df['dow'] = df['dow_code'].apply(lambda x: dow_expanded[x])

#print(df.dtypes)
#print( f.columns)

print(df.head(10))

