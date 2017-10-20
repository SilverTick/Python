
# 5 - Visualising the Data

__Table of Contents__
 * Data visualisation
    - [Pandas](#pandas)
    - [matplotlib](#matplotlib)
    - [seaborn](#seaborn)


<a id="seaborn"></a>
### seaborn

```python
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

sns.heatmap(df.isnull(), ytickslabels=False, cbar=False, cmap='viridis') #a quick visual representation of all null values

```