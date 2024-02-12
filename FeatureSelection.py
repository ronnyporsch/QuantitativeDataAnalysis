import numpy as np
import pandas as pd


# TODO this method does currently not work as expected (will delete variables that are closely correlated with sales?
def dropFeaturesWithHighCorrelation(df: pd.DataFrame, corr_matrix, threshold: float):
    """
    :source: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    :param df:
    :param corr_matrix: correlation matrix
    :param threshold: between 0 (no correlation) and 1 (max correlation)
    :return: df without highly correlated features
    """
    featuresStart = set(df.columns.to_list())
    # Create correlation matrix
    # corr_matrix = df.select_dtypes(['number']).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    newDf = df.drop(to_drop, axis=1, inplace=False)
    droppedFeatures = featuresStart.difference(set(newDf.columns.to_list()))
    print("dropped " + str(droppedFeatures) + " features")
    return newDf


def selectFeatures(df) -> pd.DataFrame:
    corr_matrix = df.corr()

    df = dropFeaturesWithHighCorrelation(df, corr_matrix, 0.8)

    corr_matrixSales = df.corr()['sales'].abs().sort_values(ascending=False)
    top_features = corr_matrixSales[1:].index
    print("selected features: " + str(top_features))
    top_features = list(top_features) + ['sales']

    return df[top_features]
