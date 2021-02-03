# work in progress
#
# pip install robin_stocks
# pip install pandas
#
# https://algotrading101.com/learn/robinhood-api-guide/
# http://www.robin-stocks.com/en/latest/quickstart.html


import robin_stocks as rs
import pandas as pd 

login = r.login(username, password)
tesla_data= rs.stocks.get_stock_historicals("TSLA", interval="day", span="year")
tesla_dataframe= pd.DataFrame(tesla_data)
