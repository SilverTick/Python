#General tips & tricks

Random formats for reference, useful or interesting tips that I find helpful/speeds things up!

Below are some formats for reference:

```python
df.apply(lambda column: column.function()) #applies a function across columns
df = pd.DataFrame() #creates an empty dataframe
```

Below are some useful snippets:

```python
np.arange(0.1,1.0,0.1) #returns an array of numbers evenly spaced at a distance of 0.1, from 0.1 to 1.0

df['col'].dropna().astype(int).value_counts() #removes na, changes type from float to integers, and returns the counts.

df['Income per Capita (,000)'] = df['Income per Capita'] // 1000 #creates a new column in thousands

```

```python
returns = pd.DataFrame() #create empty dataframe named return
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change() #creates new column, with the percentage change in Close
    
```