import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from Constants import cachedDfFilePath
from DataCleaning import cleanData
from DataReadingAndMerging import readAndMergeData
from FeatureSelection import selectFeatures
from ModelTrainingAndEvaluation import trainModel, tuneHyperParametersRF
from util import *

# models = [RandomForestClassifier(), GaussianNB(), LogisticRegression()]
models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    GaussianNB()
    # MLPClassifier()
]


def inspectData(data: pd.DataFrame):
    # printDf(data)
    # data.info()
    for column in data.columns:

        if type(data[column]) is pd.DataFrame:
            continue
        print(str(column) + ": ", end="")
        print(data[column].unique())


def trainAllModels(data: pd.DataFrame, dependentVariable: str):
    for model in models:
        trainModel(data, dependentVariable, model)


def tuneAllModels(data: pd.DataFrame, dependentVariable: str):
    tuneHyperParametersRF(data, dependentVariable)


def upsampleMinority(data: pd.DataFrame):
    # upsample minority class
    df_majority = data[(data['sales'] == 0)]
    df_minority = data[(data['sales'] == 1)]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=42)  # reproducible results
    # Combine majority class with upsampled minority class
    return pd.concat([df_minority_upsampled, df_majority])


# def executeOrReadFromCache(function, fromCache=False):
#     if fromCache:
#         return pd.read_csv(cachedDfFilePath)
#     else:
#         return function()


def buildDF() -> pd.DataFrame:
    if len(sys.argv) >= 2 and int(sys.argv[1]) >= 1:
        return pd.read_csv(cachedDfFilePath)

    inputDF = readAndMergeData()

    # inputDF = inputDF.copy().drop("Visitor_Score", axis=1)
    # transformVisitorScore(inputDF)

    inputDF = cleanData(inputDF)
    inputDF = selectFeatures(inputDF)
    inputDF = upsampleMinority(inputDF)
    inputDF.to_csv(cachedDfFilePath)
    return inputDF


if __name__ == '__main__':
    df = buildDF()
    print(df.describe())
    print(df.info)
    df.to_csv("data/output/finalDF.csv")
    # tuneHyperParametersRF(df, 'sales')
    trainAllModels(df, 'sales')
