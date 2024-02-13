import json
import re

import pandas as pd
import seaborn as sns
from IPython.display import display
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from Constants import MODELS_DIR, PLOTS_DIR


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


def createConfusionMatrix(y_test, predictions, dependentVariable, model):
    matrix = confusion_matrix(y_test, predictions)
    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, square=True, annot=True, fmt="d", cbar=True, cmap='Blues',
                xticklabels=("Not " + dependentVariable, dependentVariable),
                yticklabels=("Not " + dependentVariable, dependentVariable))
    plt.ylabel("Reality", fontsize=16)
    plt.xlabel("Prediction", fontsize=16)
    plt.title(str(model) + " with added synthetic data", fontsize=18, y=1.18)
    fig.savefig(PLOTS_DIR + "/cw_" + str(model) + ".png", dpi=300)
    plt.show()


class ModelResult:
    def __init__(self, parameters, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.parameters = parameters


def saveModelResult(model, result):
    dump(model, MODELS_DIR + "/" + str(model) + '.joblib')
    with open(MODELS_DIR + "/" + str(model) + "_results.json", 'w', encoding='utf-8') as f:
        json.dump(vars(result), f, ensure_ascii=False, indent=4)
