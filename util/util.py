import re

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from joblib import dump, load
import statsmodels.api as sm
from sklearn.preprocessing import normalize


# cleans a string by removing everything except letters and numbers
def cleanString(string: str) -> str:
    return re.sub('[^A-Za-z0-9]+', '', string)


def printDf(df: pd.DataFrame, showAllRows: bool = False):
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 10)
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
    featuresStart = len(df.columns)
    # Create correlation matrix
    corr_matrix = df.select_dtypes(['number']).corr().abs()
    # corr_matrix = df.corr().abs()
    # plt.matshow(corr_matrix)
    # plt.show()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    newDf = df.drop(to_drop, axis=1, inplace=False)
    droppedFeatures = featuresStart - len(newDf.columns)
    print("dropped " + str(droppedFeatures) + " features")
    return newDf


def createConfusionMatrix(y_test, predictions, dependentVariable, model):
    matrix = confusion_matrix(y_test, predictions)
    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, square=True, annot=True, fmt="d", cbar=True, cmap='Blues',
                xticklabels=("Not " + dependentVariable, dependentVariable),
                yticklabels=("Not " + dependentVariable, dependentVariable))
    plt.ylabel("Reality", fontsize=16)
    plt.xlabel("Prediction", fontsize=16)
    plt.title(str(model) + " with added synthetic data", fontsize=18, y=1.18)
    fig.savefig("doc/plots/cw_" + str(model) + ".png", dpi=300)
    plt.show()


# def evaluateModel(model, percentualPredictions, visitorScores):
# percentualPredictions = percentualPredictions[1:1000][:, 0]
# visitorScores = visitorScores[1:1000]
# plt.title("Line graph " + str(model))
# xsPred = [x for x in range(len(percentualPredictions))]
# xsVisScore = [x for x in range(len(visitorScores))]
# plt.plot(xsPred, percentualPredictions, color="red")
# plt.plot(xsVisScore, visitorScores, color="blue")
# plt.show()
# print("predictions: " + str(percentualPredictions))
# # for i in range(len(percentualPredictions)):
# #     print(str(model), " visitorScore: ", visitorScores[i])


def trainModel(df: pd.DataFrame, dependentVariable: str, model):
    X = df.drop([dependentVariable], axis=1)
    X = X.rename(str, axis="columns")
    y = df[dependentVariable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # visitorScoresTest = X_test['Visitor_Score'].to_numpy()
    # X_test = X_test.drop(['Visitor_Score'], axis=1)
    # X_train = X_train.drop(['Visitor_Score'], axis=1)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictPercent = model.predict_proba(X_test)
    # evaluateModel(model, predictPercent, visitorScoresTest)

    createConfusionMatrix(y_test, predictions, dependentVariable, model)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, predictions)

    # Output the accuracy
    print(str(model) + " Accuracy:", accuracy * 100, "%")
    dump(model, "models/" + str(model) + '.joblib')


    # insert an intercept
    # X['intercept'] = 1
    #
    # # Perform logistic regression without intercept
    # logit_model = sm.Logit(y, X)
    # result = logit_model.fit()
    #
    # result.summary()


def tuneHyperParametersRF(df: pd.DataFrame, dependentVariable: str):
    X = df.drop([dependentVariable], axis=1)
    X = X.rename(str, axis="columns")
    y = df[dependentVariable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    def objective(n_estimators, max_depth, min_samples_split, max_features):
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_depth=int(max_depth),
                                       min_samples_split=int(min_samples_split),
                                       max_features=min(max_features, 0.999),  # Fraction, must be <= 1.0
                                       random_state=42)

        return -1.0 * cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_squared_error").mean()

    # Bounds for hyperparameters
    param_bounds = {
        'n_estimators': (10, 250),
        'max_depth': (1, 50),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
    }

    optimizer = BayesianOptimization(f=objective, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=100)

    best_params = optimizer.max['params']
    print("best params:", best_params)
    final_model = RandomForestClassifier(n_estimators=int(best_params['n_estimators']),
                                         max_depth=int(best_params['max_depth']),
                                         min_samples_split=int(best_params['min_samples_split']),
                                         max_features=best_params['max_features'],
                                         random_state=42)
    final_model.fit(X_train, y_train)
    score = final_model.score(X_test, y_test)
    print(f"Test R^2 Score: {score}")

    best_params_formatted = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'min_samples_split': int(best_params['min_samples_split']),
        'max_features': best_params['max_features']
    }

    print("best params formatted:", best_params_formatted)
    optimized_rf = RandomForestRegressor(**best_params_formatted, random_state=42)
    optimized_rf.fit(X_train, y_train)
    predictions = final_model.predict(X_test)
    createConfusionMatrix(y_test, predictions, dependentVariable, final_model)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy * 100, "%")

