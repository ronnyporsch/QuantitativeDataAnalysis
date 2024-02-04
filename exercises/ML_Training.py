import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from util.util import *

df = pd.read_csv("inputData/Space_titanic.csv")
printDf(df)

df.info()

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="rainbow")
# Age and Cabin are a lot of NaN data


# ## Handle missing data


df["RoomService"].fillna(0, inplace=True)
df["FoodCourt"].fillna(0, inplace=True)
df["ShoppingMall"].fillna(0, inplace=True)
df["Spa"].fillna(0, inplace=True)
df["VRDeck"].fillna(0, inplace=True)
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="rainbow")

df.dropna(subset=['HomePlanet'], axis=0, inplace=True)
df.info()

# We replace the age for each class with the average age of the passenger class
for i in df.index:
    if pd.isnull(df.at[i, "Age"]):
        pclass = df.at[i, "HomePlanet"]
        pclass_1_data = df[df['HomePlanet'] == pclass]
        mean_age_pclass = pclass_1_data['Age'].mean()
        df.at[i, "Age"] = mean_age_pclass
df.info()

df.dropna(subset=['Destination'], axis=0, inplace=True)
df.info()

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="rainbow")

df.drop(["PassengerId", "Name"], axis=1, inplace=True)
df.info()

df.dropna(inplace=True)
df.info()

df

## Creat dummys and handle string columns


df["CryoSleep"] = df["CryoSleep"].astype(int)
df["VIP"] = df["VIP"].astype(int)
df["Transported"] = df["Transported"].astype(int)
df

split_data = df['Cabin'].str.split('/', expand=True, n=-1)

# Concatenate the split columns with the original DataFrame
df = pd.concat([df, split_data], axis=1)
df.rename(columns={0: "deck", 1: "num", 2: "side"}, inplace=True)
df

# df = df.loc[:,~df.columns.duplicated()]
df
# sns.countplot(x="VIP",data=df,hue="deck")
df_test = df[df["VIP"] == 1]
sns.countplot(x="deck", data=df_test)

deck = pd.get_dummies(df['deck'], drop_first=True).astype(int)
df = pd.concat([df, deck], axis=1)
df.drop(["deck"], axis=1, inplace=True)

df

df.rename(
    columns={"B": "deck_B", "C": "deck_C", "D": "deck_D", "E": "deck_E", "F": "deck_F", "G": "deck_G", "T": "deck_T"},
    inplace=True)
df

print(df["Destination"].value_counts())
# output:
# Destination
# TRAPPIST-1e      5372
# 55 Cancri e      1639
# PSO J318.5-22     725
# Name: count, dtype: int64

destination = pd.get_dummies(df['Destination'], drop_first=True).astype(int)
df = pd.concat([df, destination], axis=1)
df.drop(["Destination"], axis=1, inplace=True)

printDf(df)

df.rename(columns={"PSO J318.5-22": "pso", "TRAPPIST-1e": "trap"}, inplace=True)
df

df.info()
df.drop(["Cabin"], axis=1, inplace=True)

df.info()

planet = pd.get_dummies(df['HomePlanet'], drop_first=True).astype(int)
df = pd.concat([df, planet], axis=1)
df.drop(["HomePlanet"], axis=1, inplace=True)

df

side = pd.get_dummies(df['side'], drop_first=True).astype(int)
df = pd.concat([df, side], axis=1)
df.drop(["side", "num"], axis=1, inplace=True)

df

df.rename(columns={"S": "starboard"}, inplace=True)
df

# ## Analysis


X = df.drop(["Transported", "deck_B", "deck_C", "deck_D", "deck_E", "deck_F", "deck_G", "deck_T", "VIP"], axis=1)
y = df["Transported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train

logReg_df = LogisticRegression(max_iter=10000)
logReg_df.fit(X_train, y_train)
predictions = logReg_df.predict(X_test)

matrix = confusion_matrix(y_test, predictions)
fig = plt.figure(figsize=(8, 8))
sns.heatmap(matrix, square=True, annot=True, fmt="d", cbar=True,
            xticklabels=("Not Transported", "Transported"),
            yticklabels=("Not Transported", "Transported"))
plt.ylabel("Reality")
plt.xlabel("Prediction")

plt.show()

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)

# Output the accuracy
print("Accuracy:", accuracy * 100, "%")

# insert an intercept
df['intercept'] = 1

X = df[["intercept", "CryoSleep", 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
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

column_names = df.columns.tolist()
column_names
