import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics as m
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


def remove_outlier(df, var):
    return df[np.abs(df[var] - df[var].mean()) < 10 * df[var].std()]


def get_data():
    dataset = pd.read_csv('dc-residential-properties/DC_Properties.csv')

    df = dataset[np.isfinite(dataset['PRICE'])]
    df = df[df['QUALIFIED'] != 'U']

    # unuseful columns
    df = df.drop(['Unnamed: 0', 'GIS_LAST_MOD_DTTM', "FULLADDRESS", "CENSUS_BLOCK", "SQUARE", "CENSUS_TRACT",
                  "CENSUS_BLOCK", 'QUALIFIED'], axis=1)

    df = df.drop(
        ["ASSESSMENT_SUBNBHD", "SALEDATE", "STYLE",  "EXTWALL", "ROOF", "INTWALL", "CMPLX_NUM",
         "USECODE", "BLDG_NUM", "X", "Y",
         "LIVING_GBA", "CITY", "STATE",
         "NATIONALGRID"], axis=1)

    df = df[df["HEAT"] != 'No Data']

    num_df_list = list(df.select_dtypes(include=['float64', 'int64']))

    # for a in num_df_list:
    #     df = remove_outlier(df, a)
    df['HEAT_ISNA'] = 0 + df["HEAT"].isna()

    df = remove_outlier(df, "FIREPLACES")
    df = remove_outlier(df, "LANDAREA")
    df[['PRICE', 'ASSESSMENT_NBHD']].groupby('ASSESSMENT_NBHD').median()
    # data process
    # Some feature engineering
    df['FEAT_BEDROOMS_PER_ROOM'] = df['BEDRM'] / df['ROOMS']
    df['FEAT_STORIES_PER_GBA'] = df['STORIES'] / df['GBA']
    df['FEAT_YARD'] = df['LANDAREA'] - df['GBA']
    df['FEAT_AYB'] = df['AYB'].apply(lambda x: 2020 - x)
    df['FEAT_EYB'] = df['EYB'].apply(lambda x: 2020 - x)
    df['FEAT_GBA2'] = df['GBA'] * df['GBA']
    df['FEAT_GBA2'] = df['GBA'] * df['GBA']
    df['FEAT_CTDN'] = df['CNDTN'].map({"Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5, "Excelent": 6})
    df['FEAT_GRADE'] = df['GRADE'].map({"Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5, "Excelent": 6})


    # df.dropna(subset=['QUADRANT'],inplace= True)
    # df.dropna(subset=['FEAT_BEDROOMS_PER_ROOM'],inplace= True)
    # df.dropna(subset=['AYB'],inplace= True)
    # df.dropna(subset=['EYB'],inplace= True)

    return df


def get_model_stats(name, y_true, y_pred):
    stats = {
        "name": name,
        "EV": metrics.explained_variance_score(y_true, y_pred),
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "r2": metrics.r2_score(y_true, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "max_error": metrics.max_error(y_true, y_pred)
    }
    return stats
