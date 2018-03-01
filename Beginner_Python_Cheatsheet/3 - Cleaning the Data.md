# 3 - Cleaning the Data

__Table of Contents__
 * Data cleaning
    - [Handling null values](#null)
    - [Dealing with illogical values](#illogical)
    - [Handling outliers](#outliers)

<a id="null"></a> 
## Handling null values

I usually drop columns with more than 40% null values.

```python

null_col = list(df.columns[df.isnull().sum() > len(df)*0.4])
df.drop(null_col, axis=1, inplace=True)

```

<a id="illogical"></a> 
## Dealing with illogical values



<a id="outliers"></a> 
## Handling outliers