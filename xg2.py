import xgboost as xgb
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
y_sqrt = np.sqrt(df['PRICE'])

# X = X[:10000]
# y = y[:10000]

# Split to train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),

])

X_xgb_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', numerical_pipeline, num_df_list),
])

X_xgb = X_xgb_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_xgb = X_xgb_pipeline.transform(X_train)
X_test_xgb = X_xgb_pipeline.transform(X_test)

m = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                     max_depth=5, alpha=10, n_estimators=10)
m.fit(X_train_xgb, y_train)
y_pred_train = m.predict(X_train_xgb)
y_pred_test = m.predict(X_test_xgb)
x.append(get_model_stats('xgb', y_test, y_pred_test))

import scipy.stats as st

params = {
    "n_estimators": [1, 3, 7, 8, 10, 20, 50, 100],
    "max_depth": [1, 3, 7, 8, 10, 20],
    "learning_rate": st.uniform(0.1, 0.9),

}
xgbreg = xgb.XGBRegressor(nthread=-1, objective='reg:linear', missing=None, seed=8)

gs = RandomizedSearchCV(xgbreg, params, n_jobs=1, n_iter=10)
# gs.fit(X_train_xgb, y_train)
# gs.best_params_

m = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.16,
                     max_depth=7, alpha=10, n_estimators=20)
m.fit(X_train_xgb, y_train)
y_pred_train = m.predict(X_train_xgb)
y_pred_test = m.predict(X_test_xgb)
x.append(get_model_stats('xgb20', y_test, y_pred_test))

pd.DataFrame.from_dict(x).round(2).to_clipboard()

m = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.45,
                     max_depth=7, alpha=10, n_estimators=100)
m.fit(X_train_xgb, y_train)
y_pred_train = m.predict(X_train_xgb)
y_pred_test = m.predict(X_test_xgb)
x.append(get_model_stats('xgb100', y_test, y_pred_test))

pd.DataFrame.from_dict(x).round(2).to_clipboard()

# sqrt
X_train, X_test, y_train, y_test = train_test_split(X, y_sqrt, test_size=0.2, random_state=42)
X_xgb = X_xgb_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_xgb = X_xgb_pipeline.transform(X_train)
X_test_xgb = X_xgb_pipeline.transform(X_test)

m = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.45,
                     max_depth=7, alpha=10, n_estimators=100)
m.fit(X_train_xgb, y_train)
y_pred_train = m.predict(X_train_xgb)
y_pred_test = m.predict(X_test_xgb)
x.append(get_model_stats('xgb100sqrt', y_test, y_pred_test))

pd.DataFrame.from_dict(x).round(2).to_clipboard()

from scipy.stats import uniform

cat_df_list = list(X.select_dtypes(include=['object']))
num_df_list = list(X.select_dtypes(include=['float64', 'int64']))
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),

])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_rf_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', numerical_pipeline, num_df_list),
])

X_rf = X_rf_pipeline.fit_transform(X)
X_train_rf = X_rf_pipeline.transform(X_train)
X_test_rf = X_rf_pipeline.transform(X_test)
param_dist = {"learning_rate": uniform(0, 1),
              "gamma": uniform(0, 5),
              "max_depth": range(1, 50),
              "n_estimators": range(1, 300),
              "min_child_weight": range(1, 10)}

XGB_RS = RandomizedSearchCV(xgb.XGBRegressor(), param_distributions=param_dist, n_iter=150)
XGB_RS.fit(X_train_rf, y_train)
