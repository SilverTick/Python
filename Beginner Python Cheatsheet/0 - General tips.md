
# 0 - General tips & tricks

Random formats for reference, useful or interesting tips that I find helpful/speeds things up!

Below are some formats for reference:

```python
df.apply(lambda column: column.function()) #applies a function across columns
df = pd.DataFrame() #creates an empty dataframe
```

Below are some useful snippets:

```python

print('MAE: {0}, MSE: {1}, RMSE: {2}'.format(MAE,MSE,RMSE)) #new format for printing with variables, instead of %s or %d

np.arange(0.1,1.0,0.1) #returns an array of numbers evenly spaced at a distance of 0.1, from 0.1 to 1.0

```

General format in treating a dataset:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
%matplotlib inline #to display plots inline in jupyter notebook

```