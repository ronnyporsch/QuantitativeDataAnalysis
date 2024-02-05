# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
#
# from util.util import *
#
# df = pd.read_excel("inputData/Social_Media_Instagram.xlsx")
# df = df.fillna(0)
# printDf(df)
# corrM = df.select_dtypes(['number']).corr()
# printDf(corrM)
#
# newDf = dropFeaturesWithHighCorrelation(df, 0.8)
# printDf(newDf)
