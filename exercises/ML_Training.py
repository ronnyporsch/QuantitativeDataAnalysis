#!/usr/bin/env python
# coding: utf-8

# In[27]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("inputData/Space_titanic.csv")
df


# In[28]:


df.info()


# In[29]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ="rainbow")
# Age and Cabin are a lot of NaN data


# ## Handle missing data

# In[30]:


df["RoomService"].fillna(0,inplace=True)
df["FoodCourt"].fillna(0,inplace=True)
df["ShoppingMall"].fillna(0,inplace=True)
df["Spa"].fillna(0,inplace=True)
df["VRDeck"].fillna(0,inplace=True)
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ="rainbow")


# In[31]:


df.dropna(subset=['HomePlanet'], axis=0, inplace=True)
df.info()


# In[32]:


# We replace the age for each class with the average age of the passenger class
for i in df.index:
    if pd.isnull(df.at[i, "Age"]):
        pclass = df.at[i,"HomePlanet"]
        pclass_1_data = df[df['HomePlanet'] == pclass]
        mean_age_pclass = pclass_1_data['Age'].mean()
        df.at[i, "Age"] = mean_age_pclass
df.info()



# In[33]:


df.dropna(subset=['Destination'], axis=0, inplace=True)
df.info()


# In[34]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap ="rainbow")


# In[35]:


df.drop(["PassengerId","Name"],axis=1,inplace=True)
df.info()


# In[36]:


df.dropna(inplace=True)
df.info()


# In[37]:


df


# In[38]:


## Creat dummys and handle string columns


# In[39]:


df["CryoSleep"] =df["CryoSleep"].astype(int)
df["VIP"] =df["VIP"].astype(int)
df["Transported"] =df["Transported"].astype(int)
df


# In[40]:


split_data = df['Cabin'].str.split('/', expand=True, n=-1)

#Concatenate the split columns with the original DataFrame
df = pd.concat([df, split_data], axis=1)
df.rename(columns={0:"deck",1:"num",2:"side"},inplace=True)
df


# In[41]:


#df = df.loc[:,~df.columns.duplicated()]
df
#sns.countplot(x="VIP",data=df,hue="deck")
df_test = df[df["VIP"]==1]
sns.countplot(x="deck",data=df_test)


# In[42]:


deck = pd.get_dummies(df['deck'], drop_first=True).astype(int)
df = pd.concat([df, deck], axis=1)
df.drop(["deck"],axis=1,inplace=True)

df


# In[43]:


df.rename(columns={"B":"deck_B","C":"deck_C","D":"deck_D","E":"deck_E","F":"deck_F","G":"deck_G","T":"deck_T"},inplace=True)
df


# In[44]:


df["Destination"].value_counts()


# In[45]:


destination = pd.get_dummies(df['Destination'], drop_first=True).astype(int)
df = pd.concat([df, destination], axis=1)
df.drop(["Destination"],axis=1,inplace=True)

df


# In[46]:


df.rename(columns={"PSO J318.5-22":"pso","TRAPPIST-1e":"trap"},inplace=True)
df


# In[47]:


df.info()
df.drop(["Cabin"],axis=1,inplace=True)


# In[48]:


df.info()


# In[49]:


planet = pd.get_dummies(df['HomePlanet'], drop_first=True).astype(int)
df = pd.concat([df, planet], axis=1)
df.drop(["HomePlanet"],axis=1,inplace=True)

df


# In[50]:


side = pd.get_dummies(df['side'], drop_first=True).astype(int)
df = pd.concat([df, side], axis=1)
df.drop(["side","num"],axis=1,inplace=True)

df


# In[51]:


df.rename(columns={"S":"starboard"},inplace=True)
df


# ## Analysis

# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X = df.drop(["Transported","deck_B","deck_C","deck_D","deck_E","deck_F","deck_G","deck_T","VIP"],axis=1)
y = df["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[54]:


X_train


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


logReg_df = LogisticRegression(max_iter = 10000)
logReg_df.fit(X_train, y_train)
predictions = logReg_df.predict(X_test)


# In[57]:


from sklearn.metrics import confusion_matrix


# In[58]:


matrix = confusion_matrix(y_test, predictions)
fig = plt.figure(figsize=(8,8))
sns.heatmap(matrix,square=True,annot=True,fmt="d",cbar=True,
            xticklabels=("Not Transported","Transported"),
            yticklabels=("Not Transported","Transported"))
plt.ylabel("Reality")
plt.xlabel("Prediction")
plt.show()


# In[59]:


from sklearn.metrics import accuracy_score
# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)

# Output the accuracy
print("Accuracy:", accuracy * 100, "%")


# In[60]:


# insert an intercept
df['intercept'] = 1

X = df[["intercept","CryoSleep",'Age','VIP','RoomService','FoodCourt','ShoppingMall', 'Spa', 'VRDeck',
 'pso',
 'trap',
 'Europa',
 'Mars',
 'starboard']]


y = df['Transported']

# Perform logistic regression without intercept
logit_model = sm.Logit(y, X)
result = logit_model.fit()


result.summary()


# In[61]:


column_names = df.columns.tolist()
column_names


# In[61]:




