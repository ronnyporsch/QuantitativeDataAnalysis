import pandas as pd


def findOwner(row, df):
    if str(df['Owner_left']).casefold() != 'nan'.casefold():
        return row['Owner_left']
    else:
        return row['Owner_right']


def joinDataFrames(mainDf: pd.DataFrame, secondaryDf: pd.DataFrame) -> pd.DataFrame:
    mainDf = mainDf.join(secondaryDf.set_index('id'), on='id', how='left', lsuffix='_left', rsuffix='_right')
    mainDf['Owner_Combined'] = mainDf.apply(findOwner, args=(mainDf,), axis=1)
    mainDf = mainDf.drop('Owner_left', axis=1)
    mainDf = mainDf.drop('Owner_right', axis=1)
    return mainDf


def readAndMergeData() -> pd.DataFrame:
    mainData = pd.read_csv("data/input/CRM-Contacts_clean.csv")
    secondaryData = pd.read_csv("data/input/CRM-Calls.csv")
    return joinDataFrames(mainData, secondaryData)