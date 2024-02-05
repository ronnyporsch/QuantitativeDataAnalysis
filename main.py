import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from util.util import *

# models = [RandomForestClassifier(), GaussianNB(), LogisticRegression()]
models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(),
    GaussianNB(),
    MLPClassifier()
]


def mergePhoneColumns(data: pd.DataFrame):
    def getHighestValueFromPhoneAndNormalColumn(row, name: str):
        if name == 'phone_language_certificate.1':
            nameWithoutPhone = 'language_certificate'
        else:
            nameWithoutPhone = name.replace('phone_', '')
        # print("checking " + str(row[nameWithoutPhone]))
        if str(row[nameWithoutPhone]).casefold() != 'nan'.casefold():
            return row[nameWithoutPhone]
        else:
            # print("getting " + str(row[name]) + "from name")
            return row[name]

    columnsToRemove = []
    for (columnName, columnData) in data.items():
        if columnName.startswith(
                'phone_') and columnName != "phone_job_training_status" and columnName != "phone_pension" and columnName != "phone_problems_with_immigration_authorities" and columnName != "phone_asylum_application_year" and columnName != "phone_asylum_decision_delivered":
            data[columnName.replace('phone_', '')] = data.apply(getHighestValueFromPhoneAndNormalColumn, axis=1, name=columnName)
            columnsToRemove.append(columnName)
    for column in columnsToRemove:
        print("removing column: ", column)
        data.drop(column, axis=1, inplace=True)
        # data.drop(column.replace('phone_', ''), axis=1, inplace=True)


def inspectData(data: pd.DataFrame):
    # printDf(data)
    # data.info()
    for column in data.columns:

        if type(data[column]) is pd.DataFrame:
            continue
        print(str(column) + ": ", end="")
        print(data[column].unique())


def cleanData(data: pd.DataFrame):
    def cleanGeburtsjahr(x):
        if (len(str(x))) == 2 and x > 24:
            return 1900 + x
        if (len(str(x))) != 4:
            return 0
        return x

    def transformColumnToInt(name):
        data[name] = pd.to_numeric(data[name], errors='coerce').fillna(0).astype('int')


    data = data.replace(' ', '_', regex=True)
    mergePhoneColumns(data)
    for column in data.columns:
        # fill NaN with 0 and convert to int
        if data[column].dtype == np.float64:  # and column != "Average_Time_Spent_Minutes":
            data[column].fillna(0, inplace=True)
            data[column] = data[column].astype(int)

    data['BornInGermany'] = np.where(data['Einreisejahr'].str.lower().str.contains('deutschland'), 1, 0)
    data['Geburtsjahr'] = data['Geburtsjahr'].apply(cleanGeburtsjahr)
    columnsToInt = ['Rentenbeitraege', 'Einreisejahr', 'Number_Of_Chats', 'Kinder', 'phone_pension', 'temporary_right_of_residence_since', 'Anzahl_Kinder_unter_25_Jahre',
                    'acquired_right_of_residence']
    columnsToDummy = ['basis_for_naturalization_check', 'Familienstand', 'Nationalit_t', 'completed_job_training', 'Other_State', 'Rente', 'utm_campaign',
                      'phone_problems_with_immigration_authorities', 'Berufsausbildung_anerkannt', 'valid_national_passport_(country_of_origin)', 'language_certificate', 'Minijob',
                      'Email_Opt_Out', 'Integrationsnachweis', 'application_permanent_right_of_residence', 'Was_wollen_Sie', 'basis_for_naturalization', 'graduation',
                      'asylum_status.1']

    for i in range(len(columnsToInt)):
        transformColumnToInt(columnsToInt[i])

    for i in range(len(columnsToDummy)):
        dummies = pd.get_dummies(data[columnsToDummy[i]], drop_first=True, prefix=columnsToDummy[i]).astype(int)
        data = pd.concat([data, dummies], axis=1)

    return data
    # data['BornInGermany'] = np.nan
    #
    # for index, row in data.iterrows():
    #     if str(row['Einreisejahr']).casefold() == 'In Deutschland geboren'.casefold():
    #         print("is 1")
    #         data['BornInGermany'] = 1
    #     else:
    #         # print("is 0")
    #         data['BornInGermany'] = 0


def findUsefulVariables(data: pd.DataFrame):
    # corrM = data.select_dtypes(['number']).corr()  # only look at numbers
    data = dropFeaturesWithHighCorrelation(data, 0.8)
    data = data.select_dtypes(exclude=['object', 'bool'])
    for column in data.columns:
        # fill NaN with 0 and convert to int
        if data[column].dtype == np.int64 or data[column].dtype == np.int32:
            data[column].fillna(0, inplace=True)
            data[column] = data[column].astype(int)
    return data


def transformSalesColumn(data: pd.DataFrame) -> pd.DataFrame:
    data['sales'] = data['sales'].apply(lambda x: 0 if x == 0 else 1)
    return data


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


# def transformVisitorScore(data: pd.DataFrame):
#     data['Visitor_Score'] = pd.to_numeric(data['Visitor_Score'], errors='coerce').fillna(0).astype('int')
#     data['Visitor_Score'] = data['Visitor_Score'].apply(lambda x: 10000 if x > 10000 else x)
#
#     data['Visitor_Score'] = MinMaxScaler().fit_transform(data[['Visitor_Score']])
#     return data


def buildDF():
    def mergeOwners(row):
        if str(inputDF['Owner_left']).casefold() != 'nan'.casefold():
            return row['Owner_left']
        else:
            return row['Owner_right']

    if len(sys.argv) >= 2 and sys.argv[1].lower() == 'true':
        return pd.read_csv("data/cache/cachedDF.csv")
    else:
        mainData = pd.read_csv("data/input/CRM-Contacts_clean.csv")
        additionalData = pd.read_csv("data/input/CRM-Calls.csv")
        inputDF = mainData.copy().drop("Visitor_Score", axis=1)
        # transformVisitorScore(inputDF)
        # print(inputDF.describe())
        # print(inputDF.info)
        # exit(5)

        inputDF = inputDF.join(additionalData.set_index('id'), on='id', how='left', lsuffix='_left', rsuffix='_right')
        inputDF['Owner_Combined'] = inputDF.apply(mergeOwners, axis=1)
        inputDF = inputDF.drop('Owner_left', axis=1)
        inputDF = inputDF.drop('Owner_right', axis=1)

        inputDF = transformSalesColumn(inputDF)
        # inspectData(inputDF)
        inputDF = cleanData(inputDF)
        # inspectData(inputDF)
        # exit(5)
        inputDF = findUsefulVariables(inputDF)
        inputDF = upsampleMinority(inputDF)
        inputDF.to_csv("data/cache/cachedDF.csv")
        # printDf(inputDF)

        return inputDF


if __name__ == '__main__':
    df = buildDF()
    corr_matrix = df.corr()['sales'].abs().sort_values(ascending=False)
    print("Correlation Matrix: ")
    printDf(corr_matrix, True)
    # plt.matshow(df.corr()['sales'])
    # plt.show()

    top_features = corr_matrix[1:40].index
    print("selected features: " + str(top_features))
    top_features = list(top_features) + ['sales']

    df = df[top_features]
    # tuneHyperParametersRF(df, 'sales')
    trainAllModels(df, 'sales')
