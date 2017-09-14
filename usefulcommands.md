#connecting to sqlite (sqlalchemy)

from sqlalchemy import create_engine 

engine = create_engine('sqlite:///census.sqlite') # Create an engine that connects to the census.sqlite file on sqlite

print(engine.table_names())

---

from sqlalchemy import Table

from sqlalchemy import select

census = Table('census', metadata, autoload=True, autoload_with=engine) # Reflect the census table from the engine: census

stmt = select([census])
print(stmt)

results = connection.execute(stmt).fetchall()

first_row = results[0]

#importing data

##read csv
pd.read_csv(_, na_values='_', parse_dates=['nameofcol'])

##read excel
pd.read_excel(_, na_values='_', sheetname='_')

pd.ExcelFile('')
_.sheet_names #attribute of the excel that gives its sheetnames

##load a JSON file
with open("_.json") as json_file:
    json_data = json.load(json_file)

for k in json_data.keys():
    print(k + ': ', json_data[k]) # Print each key-value pair in json_data

##from web using DataReader
from pandas_datareader.data import DataReader #https://pandas-datareader.readthedocs.io/en/latest/
from datetime import date 
start = date(YYYY, MM, DD) #default is 2010 #if jan, just input 1, not 01
end = date(YYYY, MM, DD) #default is today
ticker = '_' #can also be series, etc (for multiple, insert as list)
data_source = 'google' #or fred, etc
stock_prices = DataReader(ticker, data_source, start, end)

if multiple tickers, store as panel (3d array). to convert back to dataframe:
panel.to_frame(). it gives multi-index.

unstacked = df['col'].unstack() #to unstack. can do it without a particular column as well with df.unstack()

##from web HTTP requests
import requests
url='http://'
resp = requests.get(url)
html_doc = resp.text #or resp.json()
print(html_doc)

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc)
pretty_soup = soup.prettify() #makes html readable
print(pretty_soup)

title = soup.title #just gets the title
a_tags = soup.find_all('a') #finds all the links
for link in a_tags:
    print(link.get('href'))

others: #scrapy is a crawler
"You give Scrapy a root URL to start crawling, then you can specify constraints on how many (number of) URLs you want to crawl and fetch,etc"

##from API
import requests
url='http://'
resp = requests.get(url)
html_doc = resp.text #or resp.json()
print(html_doc) #same as from web but replace the url with the api url.

eg the url shld be
'http://www.omdbapi.com' for the data corresponding to the movie The Social Network. The query string should have two arguments: apikey=ff21610b and t=social+network. You can combine them as follows: apikey=ff21610b&t=social+network

url='http://www.omdbapi.com/?apikey=ff21610b&t=social+network'

if json:
json_data = r.json()
for k in json_data.keys():
    print(k + ': ', json_data[k])

to extract (read from keys!)
eg:
pizza_extract = json_data['query']['pages']['24768']['extract']

##API with authentication (twitter)
import tweepy

access_token = "_"
access_token_secret = "_"
consumer_key = "_"
consumer_secret = "_" 

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)

l = MyStreamListener()
stream = tweepy.Stream(auth, l) #a class that defined tweet streem listener as MyStreamListener. have to create own?

stream.filter(track=['clinton','trump','sanders','cruz'])# Filter Twitter Streams to capture data by the keywords

###reading the data
import json
tweets_data_path='tweets.txt'

tweets_data = []

tweets_file = open(tweets_data_path, "r") # Open connection to file

for line in tweets_file:
    tweet=json.loads(line)
    tweets_data.append(tweet)

tweets_file.close() # Close connection to file

print(tweets_data[0].keys()) # Print the keys of the first tweet dict

###store data in df

df = pd.DataFrame(tweets_data, columns=['text','lang']) #see from keys previously that there were text n lang.


df_copy = df.copy()

#exploring data
_.shape
_.info
_.head()
_.tail()

##for numeric data

_.describe() #includes mean, median and std
_.mode() #use floor division operator // to add a new column for floats
_.quantile([0.25,0.75])

quantiles = np.arange(0.1,1.0,0.1)
deciles = _.quantile(quantiles)
deciles.plot(kind='bar', title='_')
plt.tight_layout()

top_5 = list['col'].nlargest(n=5) #return top 5

##for categorical data

df.col.nunique() #to get the number of unique values
df.apply(lambda column: column.nunique()) #repeats this across cols
_.value_counts() #counts true values/ how many times each unique value occurs
_.value_counts(normalize=True) #returns percentage of true values
.dropna().astype(int).value_counts() #removes na and change gype from float to int

#manipulating data
pd.concat() #adds vertically. can just put in a list.

_.columns = ['newtitle'] #rename cols

condition = list[list[colname] == 'condition'] #select rows that fulfill the condition in the column
col = list[list.colname == 'condition']

col.sort_values('colname', ascending=False) #sort by that column, descending

_.set_index('col', inplace=True) #sets the column as the index. instead of assigning result to a new variable, you can also put inplace=True.

list.loc[(list.col=='_') & (list['col']==''), 'maxcol'].idxmax #multiple conditions,take the max of maxcol and returns the index

list = _.index.tolist() #converts index to list. useful when u find the top 5 and want to convert it to list

.dropna()
.fillna()
.astype(int)
.div(1e6) #divides by 1million
df.drop('colname', axis=1) #drops a column
df.groupby('colname') #list it to see it
df.groupby('colname').'col'.mean() #groups by each category in the column, aggregates by mean on 'col'

df.groupby(['',''])
df.agg({'Average': 'mean', 'Median': 'median', 'Standard Deviation': 'std'}) #using the values, and putting the key as header

df['newcol'] = float('NaN') #create new col
df['newcol'][df['col'] < 18] = 1 #assign value to newcol 

#exclude outliers
df[df.col < df.col.quantile(.95)] #remove those above .95 percentile

#visualising data
import matplotlib.pyplot as plt
_.plot(title='_') #if multiple plots,  include subplots=True; if two plots together, can include secondary_y='' to make nicer

plt.show()

import seaborn as sns

sns.distplot('_', bins=50, kde=False, rug=True) #default is histogram, kde=True where the histogram is smoothed, rug=False which does not add markers at bottom of chart to indicate density

for column in list.columns:
    sns.distplot(list[column], hist=False, label=column) #iterate over each col and plots it



ax.axvline(_.mean(), color='b')
ax.axvline(_.median(), color='g') #assign plot to ax, then use these to highlight the mean and median as lines

.plot(kind='bar') #for horizontal barplot, use barh
plt.xticks(rotation=45) #rotates ticks to show nicely

plt.xlabel()

sns.countplot(x='_', hue='_', data=df) #x is the values for x axis, hue is the  additional split for each bar. data is the dataframe. count plot gives count for categorical data

sns.barplot(x='_', y='_', data=df, estimator=np.median) #estimator shows the chosen method in the plot. if unspecified it uses mean.

sns.pointplot(x='_', y='_', hue='_', data=df) #plot against multiple metrics. x is cat, y is numerical.

sns.boxplot(x='_', y='_', data=df)
sns.swarmplot(x='_', y='_', data=df)

df.groupby('_')

for country, data in df:
    data.plot(title=country)
    plt.show() # Iterate over df and plot per country




def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()[1:-1]
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0


##checking data and cleaning it

# Check whether the first column is 'Life expectancy'
assert df.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert df.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert df['Life expectancy'].value_counts()[0] == 1

---

# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(gapminder,id_vars='Life expectancy') #this melts data down into one column

# Convert the year column to numeric
gapminder.year = pd.to_numeric(gapminder['year'])

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtypes == np.float64

---

# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

---

# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.year).all().all()

# Drop the missing values
gapminder = gapminder.dropna()

# Print the shape of gapminder
print(gapminder.shape)



-------


#links for API
https://developers.google.com/places/web-service/details
http://docs.aws.amazon.com/AWSECommerceService/latest/DG/EX_RetrievingPriceInformation.html
https://www.reddit.com/r/webdev/comments/3wrswc/what_are_some_fun_apis_to_play_with/
https://github.com/toddmotto/public-apis


---

#intro to decision trees

import numpy as np
from sklearn import tree

train["Age"] = train["Age"].fillna(train["Age"].median()) #substitute missing values with median values, or most common category.

train["Embarked"][train["Embarked"] == "S"] = 0 #convert all categorical to numerical values

--

target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))

# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(data=my_prediction, index=PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
df = pd.read_csv('my_solution_one.csv')
print(df)