
# 2 - Exploring the Data

What are we looking out for when we are exploring our data?

First and foremost, we need our data to be complete and accurate. <br>
Hence we think about the below:<br>
1) How is the data structured - what does each row/column represent?<br>
2) Is the data complete? Look for missing data (null values)<br>
3) Is the data accurate? Look for illogical values (e.g. negative age)<br>

This is a very basic start - as you get more proficient in the Data field, you will discover that there is a lot more to data exploration, cleaning and manipulation!


__Table of Contents__
 * Data exploration
    - [Basic data structure](#structure)
    - [Data selection](#selection)
    - [Numerical data exploration](#numerical)
    - [Categorical data exploration](#categorical)

<a id="structure"></a> 
## Basic data structure

```python

df.info # returns info on shape, type of data, etc
df.head() # returns top 5 rows. for just 3 rows, use df.head(3)
df.tail() # returns bottom 5 rows - always helpful to check and make sure you don't have a 'total' row below!
df.columns # returns all column names in a list
df.index # returns index info

type(df['col'].iloc[0]) # takes one example from the column and identifies type of object in the column

df.isnull() # returns dataframe of same size, and a boolean for each value whether it is null or not
df.isnull().any() # returns a boolean for each column, whether it contains any null values or not
df.isnull().sum() # returns the number of null values in each column



```
<a id="selection"></a> 
## Data selection

```python
df.loc['2015-01-01':'2015-12-31'] # search by name of row and returns the corresponding rows. this example searches by datetime

df.iloc[0] # search by index. this returns first row
df.iloc[:,0] # this gives all rows, first column.

df.xs(level='name_of_level', key='name_of_col_in_level', axis=1) # default gets row in a multilevel dataframe. adding axis=1 takes column instead.

df[df['col'] == 'condition'] # returns rows in dataframe that fulfill the condition in 'col'
df[df['col'] == 'condition']['col2'] # returns rows in ['col2'] that fulfill the condition in 'col'

df[(df['col']>2) & (df['col']<10)] # selects and returns values that fulfill conditions - use & for multiple conditions and put () around each condition

df[(df['col'] == 1) | (df['col'] == 5)] # OR condition

```

<a id="numerical"></a> 
### Numerical data exploration

```python
df.describe() # returns count, mean, median, std, max, min, quartiles

df['col'].nlargest(5) # returns top 5 values with their index
df['col'].idxmin() # returns just the index of the min value. idxmax does same for max

```
You can also use individual commands: `df.mean()`, `df.mode()`, `df.std()` etc.

For quantiles, if you want to get more than just 4 quantiles using `df.quantile([0.25, 0.5, 0.75])`, you can use the below:

```python
import numpy as np
quantiles = np.arange(0.1,1.0,0.1) # returns an array of numbers evenly spaced at a distance of 0.1, from 0.1 to 1.0
deciles = df.quantile(quantiles)

```

<a id="categorical"></a> 
### Categorical data exploration

```python
df['col'].value_counts(dropna=False) # returns the number of times each unique value occurs. for just top 5, use df['col'].value_counts().head()

df['col'].value_counts(normalize=True) # returns the same, in percentage

df['col'].unique() # returns all the unique values

df['col'].nunique() # returns the number of unique values

df['col'].dropna().astype(int).value_counts() # removes na, changes type from float to integers, and returns the counts.

```

