import re

import numpy as np
import pandas as pd
from IPython.display import display


# cleans a string by removing everything except letters and numbers
def cleanString(string: str):
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
    :return:
    """
    # Create correlation matrix
    corr_matrix = df.select_dtypes(['number']).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    return df.drop(to_drop, axis=1, inplace=False)
