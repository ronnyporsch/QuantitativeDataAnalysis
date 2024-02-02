#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the pandas library
import pandas as pd
f500 = pd.read_csv("inputData/F500.csv", index_col=0)
f500.index.name = None
f500
# Modules, libraries or functions can generally be imported with "import".


# In[2]:


# pandas' DataFrames have a ".shape" attribute that returns a tuple representing the dimensions of each axis of the object.
f500.shape


# In[3]:


# To get a look at the first few rows of our DataFrame, we can use the DataFrame.head()
f500.head()
#f500.head(20)


# In[4]:


# Similarly, we can use the DataFrame.tail() method to reveal the last rows of our DataFrame:
f500.tail(10)


# In[5]:


# Pandas DataFrames can contain columns with multiple data types, including: integer, float, and string
# To learn about the types of each column, we can use the DataFrame.dtypes attribute
f500.dtypes


# In[6]:


# If we want an overview of all the dtypes used in our DataFrame, 
# along with its shape and other information, we can use the DataFrame.info() method. 
f500.info()


# In[7]:


# There are two types of labels in pandas: Row Labels and Column Labels.
# pandas allows us to select data using these friendly labels. The secret weapon? The DataFrame.loc[] attribute! 
f500.loc[1,"Title"]


# In[8]:


# Select a single column
Title = f500["Title"]
Title
# This shortcut for selecting a single column is super popular, and we'll use it throughout our lessons.


# In[9]:


# We discovered that when we selected just one column of our DataFrame,
# we were returned to a new pandas object: a pandas.Series object.
type(f500)
type(Title)


# As we continue our journey into data selection with pandas, keep an eye on which objects are DataFrames and which ones are Series. The functions, methods, attributes, and syntax available to us will vary depending on the type of pandas object we're working with. This distinction will help you become a master of pandas ;)!

# In[10]:


# we can also select multiple columns 
f500_new = f500[["Title", "Website"]]
# we need two brackets
f500_new
# is this a Dataframe or a Series?


# In[11]:


# also possible
f500_new_test = f500.loc[:,["Title", "Website"]]
f500_new_test


# In[12]:


# Now, let's see how to select specific columns using a slice object with labels
#f500_new = f500[["Title","Industry"]]
#f500_new
f500_new = f500.loc[:,"Title":"Industry"]
f500_new


# In[13]:


# The syntax for selecting rows from a DataFrame is the same as for columns
f500_new = f500.loc[5,:]
f500_new


# In[14]:


# selectrows
f500_new = f500.loc[5:10]
f500_new
#f500_new = f500[5:10]
#f500_new
# Notice that we don't need to use the .loc attribute or nested brackets here. 
# Also, the last element in the slice is included, unlike with regular Python slicing. Selecting rows using slices is very clean!

#f500_new = f500.loc[5:10,:]
#f500_new


# In[15]:


# Until now, we've only selected either rows or columns, but not both at the same time. Howerever, it works as we would expect: 
# by combining the syntax used for selecting rows with the syntax used for selecting columns.
# Are you up to the challenge? Let's go!
# Create a new dataframe from the rows 50 to 100 and the columns Website,Employees, Sector and Fullname.
# Create a new series of your choice.
f500_new = f500.loc[50:100,["Website","Employees","Sector","Fullname"]]
f500_new
f500_series = f500["Fullname"]
f500_series
type(f500_series)


# In[16]:


# df.loc[row_label, column_label]
# df.loc["row1":"row5", ["col1", "col3", "col7"]]
f500_new = f500.loc[5:10,"Title":"Industry"]
f500_new


# In[17]:


# Let's explore the Series.value_counts() method to see how it takes each unique non-null value in a column,
# and counts how many times it appears as a value in that column.
# First, let's select just one column from the f500 DataFrame:
sectors = f500["Sector"]


# In[18]:


sectors.value_counts()


# In[19]:


# What happens when we try using the value_counts() method directly on a DataFrame object instead of a Series object? 
f500[["Industry","Sector"]].value_counts()


# calling value_counts() on a DataFrame object look slightly different than the results we got when we called it on a Series object. This is because the DataFrame.value_counts() method returns a Series object that uses a MultiIndex instead of a single index like we got when using Series.value_counts()
# 
# In the example above, the counts returned are based on a combination of the unique non-null values found in both the sector and industry columns. For example, the combination of "Energy" in the sector column and "Utilities: Gas and Electric" in the industry column occurs 22 times in our data. However, the combination of "Materials" in the sector column and "Forest and Paper Products" in the industry column only occurs once.
# 

# In[20]:


# we can use method chaining — a way to combine multiple methods together in a single line
f500["Profits"].value_counts().loc[1768.0]


# In[21]:


# With Pandas its possible to work with vectorized operations --> operations applied to multiple data points at once
f500["Revenues"].head()
#f500["Revenues"].head()+1000
#f500["Revenues"].head()/1000


# In[22]:


# It is also possible to perform these operations with two columns
f500["Revenues"].head() - f500["Profits"].head()


# In[23]:


# pandas supports many descriptive stats methods that can help us
f500["Profits"].max()
#f500["Profits"].mean()




# Series.max()
# Series.min()
# Series.mean()
# Series.median()
# Series.mode()
# Series.sum()

# In[24]:


# We can also use these values for calculation 
f500["Profits"] - f500["Profits"].mean()


# In[25]:


#if we wanted to find the median (middle) value for the revenues and profits columns, we could use the following code
medians = f500[["Revenues", "Profits"]].median(axis=0)
medians


# In[26]:


# Next, we'll learn another method that can help us - the Series.describe() method. 
# This method tells us how many non-null values are contained in the series, 
# along with the mean, minimum, maximum, and other statistics 
f500["Profits"].describe()


# In[27]:


# Also possible for the entire dataframe
f500.describe()
#f500.max()
#f500.max(numeric_only=True)
#describe = f500.describe(include=['O']) # O not 0
#describe


# In[28]:


#One difference is that we need to manually specify if you want to see the statistics for the non-numeric columns.
#By default, DataFrame.describe() will return statistics for only numeric columns. 
#If we wanted to get just the object columns, we need to use the include=['O'] parameter:
describe = f500.describe(include=['O']) # O not 0
describe


# In[29]:


# It is possible to change the index
# Creat new column with the index values (because it was our company ranking)
f500["rank"] = f500.index

# Set 'columnName' as the new index
f500.set_index('Title', inplace=True)
f500.index.name = None
f500


# In[30]:


# This makes it easier for us to access values for the specific companies (otherwise also possible, of course).
f500.loc["Apple","Revenues"]


# In[31]:


# reset Index
f500.reset_index(inplace=True)
f500


# In[32]:


f500.set_index('index', inplace=True)
f500.index.name = None
f500


# In[33]:


# Perform assignment in pandas
top5_rank_revenue = f500[["rank", "Revenues"]].head()
print(top5_rank_revenue)


# In[34]:


top5_rank_revenue["Revenues"] = 0
print(top5_rank_revenue)


# In[35]:


# By providing labels for both axes, we can assign them to a single value within our dataframe.
top5_rank_revenue.loc["Apple", "Revenues"] = 999
print(top5_rank_revenue)


# In[36]:


# we can use boolean indexing to change all rows that meet the same criteria
num_bool = top5_rank_revenue["Revenues"]>0
result = top5_rank_revenue[num_bool]
result
#print(num_bool)


# In[37]:


# We can also combine boolean indexing and assigning
top5_rank_revenue.loc[top5_rank_revenue["Revenues"] == 999,"Revenues"] = 0
top5_rank_revenue


# In[38]:


# to remind you
top_2_sectors = f500["Sector"].value_counts().head(2)
print(top_2_sectors)


# In[39]:


# Now a task, find the 5 most frequent values for the Industry column for the Financials sector 
# and  the 10 most frequent values for the Industry column for the Energy sector 


# In[40]:


Sector_Finance = f500.loc[f500["Sector"] == "Financials", "Industry"].value_counts().head(5)
Sector_Energy = f500.loc[f500["Sector"]=="Energy","Industry"].value_counts().head(10)
Sector_Finance
#Sector_Energy

