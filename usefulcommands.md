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

unstacked = df['col'].unstack() #to unstack. can do it without a particular column as well with df.unstack()



#exploring data
_.shape
_.info
_.head()
_.tail()

##for numeric data

_.describe() #includes mean, median and std
_.mode() #use floor division operator // to add a new column for floats
_.quantile([0.25,0.75])

quantiles = np.arange(0.1,1.0,0.1)
deciles = _.quantile(quantiles)
deciles.plot(kind='bar', title='_')
plt.tight_layout()

top_5 = list['col'].nlargest(n=5) #return top 5

##for categorical data

df.col.nunique() #to get the number of unique values
df.apply(lambda column: column.nunique()) #repeats this across cols
_.value_counts() #counts true values/ how many times each unique value occurs
_.value_counts(normalize=True) #returns percentage of true values
.dropna().astype(int).value_counts() #removes na and change gype from float to int

#manipulating data
pd.concat() #adds vertically. can just put in a list.

_.columns = ['newtitle'] #rename cols

condition = list[list[colname] == 'condition'] #select rows that fulfill the condition in the column
col = list[list.colname == 'condition']

col.sort_values('colname', ascending=False) #sort by that column, descending

_.set_index('col', inplace=True) #sets the column as the index. instead of assigning result to a new variable, you can also put inplace=True.

list.loc[(list.col=='_') & (list['col']==''), 'maxcol'].idxmax #multiple conditions,take the max of maxcol and returns the index

list = _.index.tolist() #converts index to list. useful when u find the top 5 and want to convert it to list

.dropna()
.fillna()
.astype(int)
.div(1e6) #divides by 1million
df.drop('colname', axis=1) #drops a column
df.groupby('colname')
df.groupby('colname').'col'.mean() #groups by each category in the column, aggregates by mean on 'col'

#visualising data
import matplotlib.pyplot as plt
_.plot(title='_') #if multiple plots,  include subplots=True; if two plots together, can include secondary_y='' to make nicer

plt.show()

import seaborn as sns

sns.distplot('_', bins=50, kde=False, rug=True) #default is histogram, kde=True where the histogram is smoothed, rug=False which does not add markers at bottom of chart to indicate density

for column in list.columns:
    sns.distplot(list[column], hist=False, label=column) #iterate over each col and plots it



ax.axvline(_.mean(), color='b')
ax.axvline(_.median(), color='g') #assign plot to ax, then use these to highlight the mean and median as lines

.plot(kind='bar') #for horizontal barplot, use barh
plt.xticks(rotation=45) #rotates ticks to show nicely

plt.xlabel()