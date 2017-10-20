
# 2 - Exploring the Data

__Table of Contents__
 * Data exploration
    - [Basic data structure](#structure)
    - [Data selection](#selection)
    - [Numerical data exploration](#numerical)
    - [Categorical data exploration](#categorical)

<a id="structure"></a> 
## Basic data structure

```python

df.info #returns info on shape, type of data, etc
df.head() #returns top 5 rows. for just 3 rows, use df.head(3)
df.tail() #returns bottom 5 rows
df.columns #gives all column names in a list

type(df['col'].iloc[0]) #takes one example from the column and identifies type of object in the column

```
<a id="selection"></a> 
## Data selection

```python
df.loc['2015-01-01':'2015-12-31'] #search by name of row and returns the corresponding rows. this example searches by datetime

df.iloc[0] #search by index. this returns first row
df.iloc[:,0] #this gives all rows, first column.

df.xs(level='name_of_level', key='name_of_col_in_level', axis=1) #default gets row in a multilevel dataframe. adding axis=1 takes column instead.

```

<a id="numerical"></a> 
### Numerical data exploration

```python
df.describe() #returns count, mean, median, std, max, min, quartiles

df['col'].nlargest(5) #returns top 5 values with their index
df['col'].idxmin() #returns just the index of the min value. idxmax does same for max

```
You can also use individual commands: `df.mean()`, `df.mode()`, `df.std()` etc.

For quantiles, if you want to get more than just `df.quantile([0.25, 0.5, 0.75])`, you can use the below:

```python
import numpy as np
quantiles = np.arange(0.1,1.0,0.1) #returns an array of numbers evenly spaced at a distance of 0.1, from 0.1 to 1.0
deciles = df.quantile(quantiles)

```

<a id="categorical"></a> 
### Categorical data exploration

```python
df['col'].value_counts() #returns the number of times each unique value occurs. for just top 5, use df['col'].value_counts().head()

df['col'].value_counts(normalize=True) #returns the same, in percentage

df['col'].nunique() #returns the number of unique values

df['col'].dropna().astype(int).value_counts() #removes na, changes type from float to integers, and returns the counts.

```

