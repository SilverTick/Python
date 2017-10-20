
# 4 - Manipulating the Data

__Table of Contents__
 * Data structure
    - [Changing data structure](#structure)
 * Data type
    - [Changing data type](#type)
 * Code snippets 
    - [Link commands together](#link)


<a id="structure"></a> 
## Changing data structure

<a id="type"></a> 
## Changing data type

```python
float('123') #changes an integer or string to a float

import pandas as pd
pd.to_numeric(df['col']) #converts to numeric

```

<a id="link"></a> 
## Code snippets

Below are some useful snippets of code.

Creating a new column for different units

```python
df['Income per Capita (,000)'] = df['Income per Capita'] // 1000 #creates a new column in thousands

```

Breaking down datetime into individual columns

```python
import datetime
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Date'] = df['timeStamp'].apply(lambda time: time.date())

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek) #returns integers
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap) #map integers to actual day using dictionary

```

```python
import pandas as pd
returns = pd.DataFrame() #create empty dataframe named return

for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change() #creates new column, with the percentage change in Close

```