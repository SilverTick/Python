
# Getting data in

## Importing data from a file

### CSV files
```python

import pandas as pd
df = pd.read_csv('file.csv')
df

```

If your file is not in the same folder, right click the file to 'Get Info'. Copy the 'Where' location and paste it, such that instead of `'file.csv'` your file path is `'/Users/abc/Documents/untitled folder/file.csv'`.

To replace NA values and read dates correctly, use the below code instead. 

```python

import pandas as pd
df = pd.read_csv('file.csv', na_values='NAN', parse_dates=['date_column'])
df

```

Replace `'NAN'` with the current shortform for your NA values, and replace `'date_column'` with the name of your column containing dates.


### Excel files

This first method is quicker if you just want a couple of sheets from the excel file. Replace `'sheetname'` with the name of the sheet you want to import. Default is the first sheet if `sheetname` is not specified. Again you can include `na_values` to convert those.

```python

import pandas as pd
df = pd.read_excel('file.xlsx', na_values='NAN', sheetname='sheetname')

```

If you have a huge Excel file with many sheets and you need to explore first, OR if you want to loop through multiple sheets, use the below.

```python

import pandas as pd
xl = pd.ExcelFile('file.xlsx')
xl.sheetnames #returns names of all the sheets in the file
df = xl.parse

```


### JSON files

```python

import pandas as pd
df = pd.read_json('file.json')
df

```


## Connecting to servers (MSSQL) using pandas

https://gist.github.com/hunterowens/08ebbb678255f33bba94

Using SQLalchemy to create an engine to connect to SQLite/ PostgreSQL is also possible I believe, but the code seems bulkier.