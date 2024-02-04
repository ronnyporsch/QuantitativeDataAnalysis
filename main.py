from util.util import *


def mergePhoneColumns(data: pd.DataFrame):
    def getHighestValueFromPhoneAndNormalColumn(row, name: str):
        NameWithoutPhone = name.replace('phone_', '')
        if row[NameWithoutPhone] == 1 or row[name] == 1:
            return 1
        else:
            return 0

    columnsToRemove = []
    for (columnName, columnData) in data.items():
        if columnName.startswith('phone_'):
            # data['Combined_' + columnName.replace('phone_', '')] = data.apply(combinePhoneColumns, axis=1, name=columnName)
            data[columnName.replace('phone_', '')] = data.apply(getHighestValueFromPhoneAndNormalColumn, axis=1, name=columnName)
            columnsToRemove.append(columnName)
    for column in columnsToRemove:
        data.drop(column, axis=1, inplace=True)
        # data.drop(column.replace('phone_', ''), axis=1, inplace=True)


if __name__ == '__main__':
    originalDf = pd.read_excel("")  # TODO
    df = originalDf.copy().drop("Visitor_Score")

    mergePhoneColumns(df)
    printDf(df)
