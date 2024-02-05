#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math
import pylab
import scipy.stats as stats
import seaborn as sns
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv("inputData/train.csv")
df


# ## Explore the data set
# Data Dictionary
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex   	    Sex
# Age	        Age in years
# sibsp	    # of siblings / spouses aboard the Titanic
# parch	    # of parents / children aboard the Titanic
# ticket	    Ticket number
# fare	    Passenger fare
# cabin	    Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# the "cabin" column has relatively few entries
# there are also some empty entries for age


# In[6]:


df.describe()


# In[6]:





# ## Visualization

# In[7]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ="rainbow")
# Age and Cabin are a lot of NaN data


# In[8]:


# View the distribution of age
sns.displot(df["Age"])


# In[9]:


# How many survived?
sns.countplot(x="Survived",data=df)


# In[10]:


sns.countplot(x="Survived",data=df,hue="Sex")
# In relation, women survived much more often. "Women and children first".


# In[11]:


df['Pclass'] = df['Pclass'].astype(str)
sns.countplot(x="Survived",data=df,hue="Pclass")
# Passengers in 3rd class (apparently the cheaper tickets) also survived proportionally less.


# In[12]:


df['Pclass'] = df['Pclass'].astype(int)
sns.pairplot(df,hue="Survived")


# In[13]:


# For example: "Pclass" and "Fare", prices for 1st class are higher and 
# these passengers have also survived more often.
# With these plots, we can understand a lot about 
# the relationships between the variables and the effect on our dependent variable.
# We also see the distributions for the variables grouped according to the dependent variable.


# ## Clean up data

# In[14]:


df.drop("Cabin",axis=1,inplace=True)


# In[15]:


# We still need to handle the missing values in the "Age" column.
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ="rainbow")


# In[16]:


sns.boxplot(x="Pclass",y="Age",data=df)


# In[17]:


# We replace the age for each class with the average age of the passenger class
for i in df.index:
    if pd.isnull(df.at[i, "Age"]):
        pclass = df.at[i,"Pclass"]
        pclass_1_data = df[df['Pclass'] == pclass]
        mean_age_pclass = pclass_1_data['Age'].mean()
        df.at[i, "Age"] = mean_age_pclass
df


# In[18]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ="rainbow")
# looks good now


# In[19]:


df.info()
# There are still 2 missing values for Embarked


# In[20]:


df.dropna(inplace=True)


# In[21]:


df.info()


# ## Preparing data for analysis

# In[22]:


df["Sex"]
# We cannot use the categorical variables in the analysis in this way.


# In[23]:


# Conversion of the categorical variable to dummy variables
sex = pd.get_dummies(df['Sex'], drop_first=True).astype(int)
sex
# Now this variable is a dummy variable coded in 0 and 1 (1 = male, 0 = female)
# If we set drop_first = True, then the first column of the dummy variable is deleted. 
# This is not required for the analysis, 
# as 0 in the 2nd column contains the same information as 1 in this column.


# In[24]:


# Add the dummy variables to the original DataFrame
df = pd.concat([df, sex], axis=1)
df



# In[25]:


# Now we can delete the original column "Sex" and would like to rename the new column.
df.drop(["Sex"],axis=1,inplace=True)
df.rename(columns={"male":"sex"},inplace=True)
df


# In[26]:


# Your task: Which columns still need to be recoded to dummy variables? 
# Carry out the recoding.

port = pd.get_dummies(df['Embarked'], drop_first=True).astype(int)
df = pd.concat([df, port], axis=1)
df.drop(["Embarked"],axis=1,inplace=True)
df.rename(columns={"Q":"port_Q","S":"port_S"},inplace=True)
df


# In[27]:


# Now we will create a new data frame with all the variables to be included in the analysis.
df.drop(["PassengerId","Ticket","Name"],axis=1,inplace=True)
df


# In[28]:


# insert the library sklearn
from sklearn.model_selection import train_test_split


# In[29]:


# Determine what our independent variables are and what our dependent variable is.
X = df.drop("Survived",axis=1)
y = df["Survived"]


# In[30]:


# First we divide the data set into training and test
# We always need a data set with which we train our model and a data set with which we test this model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# test_size specifies how large our test portion of the data set is (33% in this case)


# In[31]:


# X_train are the independent variables of the training data set.
X_train


# ## Data analysis

# In[32]:


# Which method to estimate model? 
# We use logistic regression, since dependent 0 and 1 variable
from sklearn.linear_model import LogisticRegression


# In[33]:


logReg_df = LogisticRegression(max_iter = 10000)
logReg_df.fit(X_train, y_train)


# In[34]:


predictions = logReg_df.predict(X_test)


# In[35]:


predictions


# In[36]:


#X_test["predictions"]=predictions
#X_test["survived"]=y_test
X_test


# In[37]:


# How well did the model perform?


# In[38]:


from sklearn.metrics import confusion_matrix


# In[39]:


# matrix = confusion_matrix(y_test, predictions)
# fig = plt.figure(figsize=(8,8))
# sns.heatmap(matrix,square=True,annot=True,fmt="d",cbar=True,
#             xticklabels=("Not survived","Survived"),
#             yticklabels=("Not survived","Survived"))
# plt.ylabel("Reality")
# plt.xlabel("Prediction")
# plt.show()


# In[40]:


from sklearn.metrics import accuracy_score
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)

# Output the accuracy
print("Accuracy:", accuracy * 100, "%")


# In[40]:





# In[41]:


# Trying to find a better model


# In[42]:


df


# In[43]:


# insert an intercept
df['intercept'] = 1

X = df[['intercept', "Pclass","Age","SibSp","Parch","Fare","sex","port_Q","port_S"]]


y = df['Survived']

# Perform logistic regression without intercept
logit_model = sm.Logit(y, X)
result = logit_model.fit()


result.summary()


# In[44]:


df


# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:





# In[44]:




