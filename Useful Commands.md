__The Beginner's Python Cheatsheet!__

This is a reference list of basic Python code, mostly in the context of Data Science and (just a little) Machine Learning.

Where I've put in links, these guys have already done a great job of it, so I'll leave it to the experts - just click into it!

Most of this will be quite basic, likely most valuable for beginner individuals just doing a quick Google to figure something out. This list is not meant to be exhaustive, just sufficient for basic purposes.

Standard terms used below - replace with your own:
- _df_ for dataframes
- _file_ for filenames
- _database_ for databases

I try to use pandas where possible, just because I really like how clean it is :)

I am also just still learning, so if there are any areas of improvement, please do share.
Enjoy!

---

# Getting data in

## Importing data from a file

### CSV files
```python

import pandas as pd
df = pd.read_csv('file.csv')
df

```

If your file is not in the same folder, right click the file to 'Get Info'. Copy the 'Where' location and paste it, such that instead of `'file.csv'` your file path is `'/Users/abc/Documents/untitled folder/file.csv'`

To replace NA values and read dates correctly, use the below code instead. 

```python

import pandas as pd
df = pd.read_csv('file.csv', na_values='NAN', parse_dates=['date_column'])
df

```

Replace NAN with the current shortform for your NA values, and replace date_column with the name of your column containing dates.


### Excel files

This first method is quicker if you just want a couple of sheets from the excel file. Replace sheetname with the name of the sheet you want to import. Default is the first sheet if sheetname is not specified.
Again you can include na_values to convert those.

```python

import pandas as pd
pd.read_excel('file.xlsx', na_values='NAN', sheetname='sheetname')

```

If you have a huge Excel file with many sheets and you need to explore first, OR if you want to loop through multiple sheets, use the below.

```python

import pandas as pd
xl = pd.ExcelFile('file.xlsx')
xl.sheetnames #returns the names of all the sheets in the file
df = xl.parse

```

##load a JSON file
with open("_.json") as json_file:
    json_data = json.load(json_file)

for k in json_data.keys():
    print(k + ': ', json_data[k]) # Print each key-value pair in json_data


## Connecting to servers using pandas

https://gist.github.com/hunterowens/08ebbb678255f33bba94

Using SQLalchemy to create an engine to connect to SQLite/ PostgreSQL is also possible I believe, but the code seems bulkier.