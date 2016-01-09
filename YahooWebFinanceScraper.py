#Scrapes stock data from Yahoo Finance
import datetime
import pandas as pd
import csv
from pandas import DataFrame
from pandas.io.data import DataReader
#NYSE TOP 40 Stocks
symbols_list = ['AAPL', 'TSLA', 'YHOO','MSFT','ALTR','WDC','KLAC', 'BAC', 'KMI' ,'SUNE', 'HPQ', 'FCX', 'GE', 'PBR', 'BABA', 'ITUB', 'XOM', 'C', 'EMC', 'MPLX', 'CNX' ,'NRG', 'S', 'EPD', 'WMT', 'ORCL']
#120MB of test data for classifiers! 
symbols=[]
for ticker in symbols_list:
    print ticker
    i,j = 1,1
    for i in range (1,13):
        print i
        for j in range(1,21):
            print j
            r = DataReader(ticker, "yahoo", start=datetime.datetime(2014, i, j))
            # add a symbol column
            r['Symbol'] = ticker 
            symbols.append(r)
            j += 1

        i += 1
# concatenate all the dataframes
df = pd.concat(symbols)
# create an organized cell from the new dataframe
cell= df[['Symbol','Open','High','Low','Adj Close','Volume']]

cell.reset_index().sort(['Symbol', 'Date'], ascending=[1,0]).set_index('Symbol').to_csv('stock.csv', date_format='%d/%m/%Y')
print "Finished writing"

