#importing data

##read csv
pd.read_csv(_, na_values='_', parse_dates=['nameofcol'])

##read excel
pd.read_excel(_, na_values='_', sheetname='_')

pd.ExcelFile('')
_.sheet_names #attribute of the excel that gives its sheetnames

##from web using DataReader
from pandas_datareader.data import DataReader #https://pandas-datareader.readthedocs.io/en/latest/
from datetime import date 
start = date(YYYY, MM, DD) #default is 2010 #if jan, just input 1, not 01
end = date(YYYY, MM, DD) #default is today
ticker = '_' #can also be series, etc (for multiple, insert as list)
data_source = 'google' #or fred, etc
stock_prices = DataReader(ticker, data_source, start, end)

if multiple tickers, store as panel (3d array). to convert back to dataframe:
panel.to_frame(). it gives multi-index.

to unstack it:
unstacked = data['col'].unstack()



#exploring data
_.shape
_.info
_.head()
_.tail()
_.value_counts() #counts true values
_.value_counts(normalize=True) #returns percentage of true values
_.mean()
_.median()
_.mode()
_.std()


#manipulating data
pd.concat #adds vertically
_.columns = ['newtitle'] #rename cols

condition = list[list[colname] == 'condition'] #select rows that fulfill the condition in the column
col = list[list.colname == 'condition']

col.sort_values('colname', ascending=False) #sort by that column, descending

_.set_index('col', inplace=True) #sets the column as the index. instead of assigning result to a new variable, you can also put inplace=True.

list.loc[(list.col=='_') & (list['col']==''), 'maxcol'].idxmax #multiple conditions,take the max of maxcol and returns the index

top_5 = list['col'].nlargest(n=5) #return top 5

list = top_5.index.tolist() #converts index to list

#visualising data
import matplotlib.pyplot as plt
_.plot(title=) #if multiple plots,  include subplots=True; if two plots together, can include secondary_y='' to make nicer

plt.show()
