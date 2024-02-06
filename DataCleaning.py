import numpy as np
import pandas as pd


def transformSalesColumn(df: pd.DataFrame) -> pd.DataFrame:
    df['sales'] = df['sales'].apply(lambda x: 0 if x == 0 else 1)
    return df


def convertVariablesToIntOrDrop(df: pd.DataFrame):
    # corrM = data.select_dtypes(['number']).corr()  # only look at numbers
    # df = df.select_dtypes(exclude=['object', 'bool'])  # TODO dont drop booleans
    df = df.select_dtypes(exclude=['object'])  # TODO dont drop booleans
    for column in df.columns:
        # fill NaN with 0 and convert to int
        # if df[column].dtype == np.int64 or df[column].dtype == np.int32:
        df[column] = df[column].astype(int)
        df[column].fillna(0, inplace=True)
    return df


def mergePhoneColumns(df: pd.DataFrame):
    def getHighestValueFromPhoneAndNormalColumn(row, name: str):
        if name == 'phone_language_certificate.1':
            nameWithoutPhone = 'language_certificate'
        else:
            nameWithoutPhone = name.replace('phone_', '')
        # print("checking " + str(row[nameWithoutPhone]))
        if str(row[nameWithoutPhone]).casefold() != 'nan'.casefold():
            return row[nameWithoutPhone]
        else:
            return row[name]

    columnsToRemove = []
    for (columnName, columnData) in df.items():
        if (columnName.startswith(
                'phone_') and columnName != "phone_job_training_status" and columnName != "phone_pension" and
                columnName != "phone_problems_with_immigration_authorities" and columnName != "phone_asylum_application_year" and
                columnName != "phone_asylum_decision_delivered"):
            df[columnName.replace('phone_', '')] = df.apply(getHighestValueFromPhoneAndNormalColumn, axis=1, name=columnName)
            columnsToRemove.append(columnName)
    for column in columnsToRemove:
        print("removing column: ", column)
        df.drop(column, axis=1, inplace=True)


def transformColumnToInt(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df[name] = pd.to_numeric(df[name], errors='coerce').fillna(0).astype('int')
    return df


def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    def cleanGeburtsjahr(year):
        if (len(str(year))) == 2 and year > 24:
            return 1900 + year
        if (len(str(year))) != 4:
            return 0
        return year

    df = df.replace(' ', '_', regex=True)
    mergePhoneColumns(df)
    for column in df.columns:
        # fill NaN with 0 and convert to int
        if df[column].dtype == np.float64:  # and column != "Average_Time_Spent_Minutes":
            df[column].fillna(0, inplace=True)
            df[column] = df[column].astype(int)

    df['BornInGermany'] = np.where(df['Einreisejahr'].str.lower().str.contains('deutschland'), 1, 0)
    df['Geburtsjahr'] = df['Geburtsjahr'].apply(cleanGeburtsjahr)
    columnsToInt = ['Rentenbeitraege', 'Einreisejahr', 'Number_Of_Chats', 'Kinder', 'phone_pension', 'temporary_right_of_residence_since', 'Anzahl_Kinder_unter_25_Jahre',
                    'acquired_right_of_residence']
    columnsToDummy = ['basis_for_naturalization_check', 'Familienstand', 'Nationalit_t', 'completed_job_training', 'Other_State', 'Rente', 'utm_campaign',
                      'phone_problems_with_immigration_authorities', 'Berufsausbildung_anerkannt', 'valid_national_passport_(country_of_origin)', 'language_certificate', 'Minijob',
                      'Email_Opt_Out', 'Integrationsnachweis', 'application_permanent_right_of_residence', 'Was_wollen_Sie', 'basis_for_naturalization', 'graduation',
                      'asylum_status.1']

    for i in range(len(columnsToInt)):
        df = transformColumnToInt(df, columnsToInt[i])

    for i in range(len(columnsToDummy)):
        dummies = pd.get_dummies(df[columnsToDummy[i]], drop_first=True, prefix=columnsToDummy[i]).astype(int)
        df = pd.concat([df, dummies], axis=1)

    df = transformSalesColumn(df)
    df = convertVariablesToIntOrDrop(df)
    return df
