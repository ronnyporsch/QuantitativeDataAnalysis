# Understand data
- `df.info()`
- `df.describe()`
- Distribution of a column: `sns.displot(df["Age"])`
- heatmap to show null/NaN/ values `sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="rainbow")`
- Correlation Matrix:
```corrM = df.select_dtypes(['number']).corr()
corrM = df.select_dtypes(['number']).corr() # only look at numbers
printDf(corrM) # print matrix
plt.matshow(corrM) # plot matrix
plt.show()
```
- BarGraph for counting category values: `sns.countplot(x="Survived",data=df,hue="Sex")` 

# Clean Data
- fill NaN data: `df["FoodCourt"].fillna(0, inplace=True)`
- boolean to int: `df["CryoSleep"] =df["CryoSleep"].astype(int)`
# Feature Engineering
- Weekdays -> isWeekend
# Visualize Data
- scatter plots
- PairPlot `sns.pairplot(df,hue="Survived")`
- BoxPlot `sns.boxplot(x="Pclass",y="Age",data=df)`
# Regression Analysis
- Measurement of the influence of one or more variables on another variable
- Prediction of a variable by one or more other variables

## Forms of regression analysis
### Simple Linear
- one independent variable infers the dependent variable
- independent variables can be metric, ordinal, nominal
### Multiple Linear
- multiple independent variables predict dependent variable
- independent variables can be metric, ordinal, nominal
### Logistic
- use when dependent variable is categorical (nominal or ordinal) (if more than two categories -> use Dummy variables (see ML_Training.py line 87-99 and ML_solution.py line 193ff)
- independent variables can be metric, ordinal, nominal

## GridSearch, Bayesian Optimization, RandomSearch

# Sources
- [Regression Analysis: An introduction to Linear and Logistic Regression (YouTube)](https://www.youtube.com/watch?v=FLJ0yYetywE)