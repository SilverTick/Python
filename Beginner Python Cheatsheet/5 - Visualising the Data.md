
# 5 - Visualising the Data

__Table of Contents__
 * Data visualisation
    - [Pandas](#pandas)
    - [matplotlib](#matplotlib)
    - [seaborn](#seaborn)

<a id="pandas"></a>
### pandas

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline #for jupyter notebook to display plots automatically without plt.show()
plt.style.use('ggplot') #displays plots in ggplot style

df['col'].plot() #single line graph with index as x
df.plot(x='col1', y='col2') #single line graph

df['col'].plot.hist(bins=10) #histogram (df.hist() also works). bins is number of 'bars' (intervals)
df[['col1','col2']].plot.box() #box plots
df.plot.scatter(x='col1', y='col2') #scatter plot
df.plot.density() #kde
df.plot.area() #area under graph

```

To customise the aesthetics (tab+shift for full details): e.g. 

`df.plot.scatter(x='col1',y='col2', color='red', edgecolor='black', lw=1, s=50, figsize=(12,3))`

    figsize - size of figure e.g. `figsize=(12,3)`
    alpha - transparency (0 to 1) e.g.`alpha=0.5`
    lw - linewidth (integers) e.g. `lw=3`
    ls - linestyle ('-', '--', ':', etc) e.g.`ls='--'`
    color - insert 'b' for blue, 'r' for red, etc or rgb hexcode (#000000). e.g. `color='red'`
    edgecolor - ('black', etc) e.g. `edgecolor='black'`
    marker - 'o', '*', '+', etc
others include: markerfacecolor, markeredgewidth, markeredgecolor, etc

To remove overlaps, use `plt.tight_layout()` - automatic reshuffling to minimize overlaps

To move the legend - append `.legend()` to the end of the code e.g. 

`df.hist().legend(bbox_to_anchor=(1,1))` or `df.hist().legend(loc=0)`
    - use strings (such as 'upper right') or, location code (integers)  (tab+shift for full details)
    - outside of the graph - use bbox_to_anchor=(1,1)


To label all lines at one go, use `plt.legend(legend_list)` where `legend_list = ['line1', 'line2', 'line3']`

To label the axes and title the graph:
    plt.xlabel('X label')
    plt.ylabel('Y label')
    plt.title('Title')

To export the image, put `plt.savefig('picture.png', dpi=200)` instead of `plt.show()` to save as a picture.

<a id="matplotlib"></a>
### matplotlib

```python
import matplotlib.pyplot as plt
%matplotlib inline #for jupyter notebook to display plots automatically without plt.show()

plt.plot(x,y) #single line graph
plt.plot(df['col1'], df['col2']) #single line graph - col1 on x, col2 on y
plt.plot(df) #plots entire df (can be multiple lines)

```
Aesthetics is similar as above in pandas (which is based off of matplotlib)
e.g. `plt.plot(x,y, label = 'line one', color='#000000', lw=2, alpha=0.5, linestyle='- .', marker='o', markersize=5)`

Minor tweaks in certain code:

```python
fig = plt.figure(figsize=(8,2)) #figsize has to be set before plotting
plt.legend(loc='upper right') #set legend separately

```

Plotting more than one figure

```python
fig = plt.figure(figsize=(10,8)) #affects size of all figures
ax1 = fig.add_axes([0,0,0.5,0.5]) #specify where the axes are positioned
ax2 = fig.add_axes([0.5,0.5,0.5,0.5]) 

ax1.plot(x1,y1, label = 'line one') #to plot the line
ax1.xlabel('x1 label')
ax1.ylabel('y1 label')
ax1.title('ax1')

ax2.plot(x2,y2, label = 'line two')
ax2.xlabel('x2 label')
ax2.ylabel('y2 label')
ax2.title('ax2')

```

<a id="seaborn"></a>
### seaborn

```python
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

sns.heatmap(df.isnull(), ytickslabels=False, cbar=False, cmap='viridis') #a quick visual representation of all null values

```