import re

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm


# cleans a string by removing everything except letters and numbers
def cleanString(string: str) -> str:
    return re.sub('[^A-Za-z0-9]+', '', string)


def printDf(df: pd.DataFrame, showAllRows: bool = False):
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    if showAllRows:
        pd.set_option('display.max_rows', None)
    display(df)


def dropFeaturesWithHighCorrelation(df: pd.DataFrame, threshold: float):
    """
    :source: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    :param df:
    :param threshold: between 0 (no correlation) and 1 (max correlation)
    :return: df without highly correlated features
    """
    # Create correlation matrix
    corr_matrix = df.select_dtypes(['number']).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    return df.drop(to_drop, axis=1, inplace=False)


def createConfusionMatrix(y_test, predictions, dependentVariable):
    matrix = confusion_matrix(y_test, predictions)
    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, square=True, annot=True, fmt="d", cbar=True,
                xticklabels=("Not " + dependentVariable, dependentVariable),
                yticklabels=("Not " + dependentVariable, dependentVariable))
    plt.ylabel("Reality")
    plt.xlabel("Prediction")
    plt.show()


def performLogisticRegression(df: pd.DataFrame, dependentVariable: str):
    X = df.drop([dependentVariable], axis=1)
    y = df[dependentVariable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    logReg_df = LogisticRegression(max_iter=10000)
    logReg_df.fit(X_train, y_train)
    predictions = logReg_df.predict(X_test)

    createConfusionMatrix(y_test, predictions, dependentVariable)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, predictions)

    # Output the accuracy
    print("Accuracy:", accuracy * 100, "%")

    # insert an intercept
    X['intercept'] = 1

    # Perform logistic regression without intercept
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    result.summary()


def tuneHyperParameters(df: pd.DataFrame, dependentVariable: str):
    x = df.drop([dependentVariable], axis=1)
    y = df[dependentVariable]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(x_train, y_train)

    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)
