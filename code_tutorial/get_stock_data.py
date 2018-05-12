# for Python 3 on Windows
# Get Stock Data
# from a CSV file

# Based on the documentation at:
# http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb

# using data from:
# https://www.nasdaq.com/symbol/amd/historical

# import matplotlib.pyplot as plt
import pandas as pd 

Location = r'C:\Data\code\anntt\code_tutorial\HistoricalQuotes.csv'
df = pd.read_csv(Location)

#print(df.ix['2018-01-04'])
print(df.head(10))
