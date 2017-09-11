##read csv
pd.read_csv(_, na_values='_', parse_dates=['nameofcol'])

##read excel
pd.read_excel(_, na_values='_', sheetname='_')

pd.ExcelFile('')
_.sheet_names #attribute of the excel that gives its sheetnames

##exploring data
_.shape
_.info
_.head()
_.value_counts() #counts true values
_.value_counts(normalize=True) #returns percentage of true values

##manipulating
pd.concat #adds vertically
