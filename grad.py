import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics as m
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

from util import get_model_stats
from util import get_data


df = get_data()
x = []

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df_list.remove('PRICE')

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),

])

# pipeline for categorical features
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder()),
])

X_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', num_pipeline, num_df_list),
])

X = df.drop(columns=['PRICE'])
y = df['PRICE']


def get_grad_model(name, XX, yy, estimators=1):
    cat_df_list = list(XX.select_dtypes(include=['object']))
    num_df_list = list(XX.select_dtypes(include=['float64', 'int64']))
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),

    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot', OneHotEncoder()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2, random_state=42)
    X_rf_pipeline = ColumnTransformer([
        ('cat', cat_pipeline, cat_df_list),
        ('num', numerical_pipeline, num_df_list),
    ])

    X_rf = X_rf_pipeline.fit_transform(XX)
    X_train_rf = X_rf_pipeline.transform(X_train)
    X_test_rf = X_rf_pipeline.transform(X_test)

    m = lin_reg = GradientBoostingRegressor(n_estimators=400, random_state=21)
    m.fit(X_train_rf, y_train)
    y_pred_test = m.predict(X_test_rf)
    return get_model_stats(name, y_test, y_pred_test)


modelStats = get_grad_model('lin_simple', X, y, 100)
x.append(modelStats)

modelStats = get_grad_model('lin_log10', X, np.log10(y), 100)
x.append(modelStats)
modelStats = get_grad_model('lin_sqrt', X, np.sqrt(y), 400)
x.append(modelStats)
modelStats = get_grad_model('xg_400_log', X, np.log(y), 400)
x.append(modelStats)

comb = [
    ["LANDAREA", "BEDRM", "ROOMS", "BATHRM", "HF_BATHRM", "AYB", "EYB", "FIREPLACES", "STORIES",
     'FEAT_BEDROOMS_PER_ROOM', 'FEAT_STORIES_PER_GBA', 'FEAT_YARD', 'FEAT_AYB', 'FEAT_EYB'],

    ["BATHRM", "HF_BATHRM", "AYB", "EYB", "FIREPLACES", "STORIES", 'FEAT_BEDROOMS_PER_ROOM', 'FEAT_STORIES_PER_GBA',
     'FEAT_YARD', 'FEAT_AYB', 'FEAT_EYB'],

    ["BATHRM", "HF_BATHRM", "ROOMS", "AYB", "EYB", "STORIES", "CNDTN", "QUADRANT", 'FEAT_BEDROOMS_PER_ROOM',
     'FEAT_STORIES_PER_GBA', 'FEAT_YARD', 'FEAT_AYB', 'FEAT_EYB'],

    ["BATHRM", "HF_BATHRM", "ROOMS", "AYB", "STORIES", "CNDTN", "WARD", 'FEAT_BEDROOMS_PER_ROOM',
     'FEAT_STORIES_PER_GBA', 'FEAT_YARD', 'FEAT_AYB', 'FEAT_EYB'],

    ["BATHRM", "HF_BATHRM", "ROOMS", "AYB", "EYB", "STORIES", "CNDTN", "ZIPCODE", 'FEAT_BEDROOMS_PER_ROOM',
     'FEAT_STORIES_PER_GBA', 'FEAT_YARD', 'FEAT_AYB', 'FEAT_EYB'],

    ["BATHRM", "HF_BATHRM", "ROOMS", "AYB", "EYB", "STORIES", "CNDTN", "ASSESSMENT_NBHD", 'FEAT_BEDROOMS_PER_ROOM',
     'FEAT_STORIES_PER_GBA', 'FEAT_YARD', 'FEAT_AYB', 'FEAT_EYB']
]

for i in range(len(comb)):
    select = comb[i]
    modelStats = get_grad_model('grad_comb' + str(i), X[select], y, 400)
    x.append(modelStats)

pd.DataFrame.from_dict(x).round(2).to_clipboard()
