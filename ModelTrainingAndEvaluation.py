import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from util import createConfusionMatrix, ModelResult, saveModelResult


# def trainAllModels(df: pd.DataFrame, dependantVariable: str):


def trainModel(df: pd.DataFrame, dependentVariable: str, model):
    X = df.drop([dependentVariable], axis=1)
    # X = X.rename(str, axis="columns")
    y = df[dependentVariable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    visitorScoresTest = transformVisitorScore(X_test['Visitor_Score'].to_numpy())
    X_test = X_test.drop(['Visitor_Score'], axis=1)
    X_train = X_train.drop(['Visitor_Score'], axis=1)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictPercent = model.predict_proba(X_test)
    # evaluateModel(model, predictPercent, visitorScoresTest)

    createConfusionMatrix(y_test, predictions, dependentVariable, model)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, predictions)
    # accuracyVisScore = accuracy_score(y_test, visitorScoresTest)
    # print("Visitor Score Accuracy:", accuracyVisScore * 100, "%")
    # createConfusionMatrix(y_test, visitorScoresTest, dependentVariable, "Visitor Score")

    # Output the accuracy
    print(str(model) + " Accuracy:", accuracy * 100, "%")

    result = ModelResult(df.columns.to_list(), accuracy)
    saveModelResult(model, result)

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


def evaluateModel(model, percentualPredictions, visitorScores):
    for i in range(len(percentualPredictions)):
        print(str(model), " visitorScore: ", visitorScores[i])


def transformVisitorScore(arr: np.array):
    # df['Visitor_Score'] = pd.to_numeric(df['Visitor_Score'], errors='coerce').fillna(0).astype('int')
    # df['Visitor_Score'] = df['Visitor_Score'].apply(lambda x: 10000 if x > 10000 else x)
    #
    # df['Visitor_Score'] = MinMaxScaler().fit_transform(df[['Visitor_Score']])
    # arr['Visitor_Score'] = arr['Visitor_Score'].replace(-np.inf, 0)
    #
    # arr['Visitor_Score'] = np.log(np.log(arr['Visitor_Score'] + 1))
    # arr['Visitor_Score'] = arr['Visitor_Score'].replace(-np.inf, 0)
    # scaled = MinMaxScaler().fit_transform(arr[['Visitor_Score']])

    arr = np.where(np.isinf(arr), 0, arr)
    arr = np.log(np.log(arr + 1) + 1)
    arr = np.where(np.isinf(arr), 0, arr)
    scaled = MinMaxScaler().fit_transform(arr.reshape(-1, 1))
    scaled[scaled < .5] = 0
    scaled[scaled >= .5] = 1

    return scaled
