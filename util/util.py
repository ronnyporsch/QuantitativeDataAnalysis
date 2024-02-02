import re

import pandas as pd
from IPython.display import display


# cleans a string by removing everything except letters and numbers
def cleanString(string):
    return re.sub('[^A-Za-z0-9]+', '', string)


def printDF(df, showAllRows=False):
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    if showAllRows:
        pd.set_option('display.max_rows', None)
    display(df)
