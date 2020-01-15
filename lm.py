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
X_numeric = X[num_df_list]
y = df['PRICE']

# Split to train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prep data with pipeline
X_prepared = X_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_prepared = X_pipeline.transform(X_train)
X_test_prepared = X_pipeline.transform(X_test)

# Regresie lineara

from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

preds = lin_reg.predict(X_train_prepared)
preds_test = lin_reg.predict(X_test_prepared)
x.append(get_model_stats('linear_regression', y_test, preds_test))

from sklearn.preprocessing import PolynomialFeatures



# pipeline for numerical features turned to polynomials
poly_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
  #  ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('std_scaler', StandardScaler()),

])

# pipeline for categorical features
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder()),
])

# transformer for polynomial features
X_poly_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', poly_pipeline, num_df_list),
])


# gradient boost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_gb_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', poly_pipeline, num_df_list),
])

X_gb = X_gb_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_gb = X_gb_pipeline.transform(X_train)
X_test_gb = X_gb_pipeline.transform(X_test)
model = GradientBoostingRegressor(n_estimators=400, random_state=21)
kfold = KFold(n_splits=10, random_state=21)

model.fit(X_train_gb, y_train)
preds = model.predict(X_train_gb)
x.append(get_model_stats('grad_boost', y_train, preds))

print("Random Search")

loss = ['ls', 'lad', 'huber']
n_estimators = [300, 500, 600]
max_depth = [10, 15]
min_samples_leaf = [1, 2, 4, 6, 8]
min_samples_split = [2, 4, 6, 10]
max_features = ['auto', 'sqrt', 'log2', None]
# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}

model = GradientBoostingRegressor(n_estimators=400, random_state=21)
rs = RandomizedSearchCV(estimator=model,
                        param_distributions=hyperparameter_grid,
                        cv=4, n_iter=50,
                        scoring='neg_mean_absolute_error', n_jobs=4,
                        verbose=5,
                        return_train_score=True,
                        random_state=42)
# rs.fit(X_train_gb, y_train)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

pd.DataFrame.from_dict(x).round(2).to_clipboard()
