import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('dc-residential-properties/DC_Properties.csv')

df = dataset[np.isfinite(dataset['PRICE'])]
# unuseful columns
df = df.drop(['Unnamed: 0', 'GIS_LAST_MOD_DTTM', "FULLADDRESS", "CENSUS_BLOCK", "SQUARE", "CENSUS_TRACT",
              "CENSUS_BLOCK"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# eda
df_missing = df.copy()
missing = df_missing.isnull().sum()
missing
np.round(100 * missing[missing > 0] / df.__len__(), 3).to_clipboard()

df = df.drop(
    ["ASSESSMENT_SUBNBHD", "SALEDATE", "STYLE", "GRADE", "CNDTN", "EXTWALL", "ROOF", "INTWALL", "CMPLX_NUM",
     "LIVING_GBA", "CITY", "STATE",
     "NATIONALGRID"], axis=1)

df = df[df["PRICE"] < 800000]

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))

sns.pairplot(df[["PRICE","LANDAREA",  "ROOMS", "BATHRM", "AYB", "EYB"]])
plt.show()