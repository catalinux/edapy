import pandas as pd
import numpy as np
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

dataset = pd.read_csv('dc-residential-properties/DC_Properties.csv')

df = dataset[np.isfinite(dataset['PRICE'])]
# unuseful columns
df = df.drop(['Unnamed: 0', 'GIS_LAST_MOD_DTTM', "FULLADDRESS", "CENSUS_BLOCK", "SQUARE", "CENSUS_TRACT",
              "CENSUS_BLOCK"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

df = df.drop(
    ["ASSESSMENT_SUBNBHD", "SALEDATE", "STYLE", "GRADE", "CNDTN", "EXTWALL", "ROOF", "INTWALL", "CMPLX_NUM",
     "LIVING_GBA", "CITY", "STATE", "X", "Y",
     "NATIONALGRID"], axis=1)

df = df[df["HEAT"] != 'No Data']
df = df[df["PRICE"] < 1200000]

plt.show()

columns = pd.DataFrame({'column_names': df.columns, 'datatypes': df.dtypes})
columns.sort_values(by="datatypes")


def remove_outlier(var):
    return df[np.abs(df[var] - df[var].mean()) < 5 * df[var].std()]


df = df[df['QUALIFIED'] != 'U']

df = remove_outlier("FIREPLACES")
df = remove_outlier("LANDAREA")

# data process
# Some feature engineering
df['FEAT_BEDROOMS_PER_ROOM'] = df['BEDRM'] / df['ROOMS']
df['FEAT_STORIES_PER_GBA'] = df['STORIES'] / df['GBA']
df['FEAT_YARD'] = df['LANDAREA'] - df['GBA']
df['FEAT_AYB'] = df['AYB'].apply(lambda x: 2020 - x)
df['FEAT_EYB'] = df['EYB'].apply(lambda x: 2020 - x)


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


x = []

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df_list.remove('PRICE')
# num_df_list.remove('PRICE')

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

x.append(get_model_stats('linear_regression', y_train, preds))



from sklearn.preprocessing import PolynomialFeatures

# pipeline for numerical features turned to polynomials
poly_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    #    ('poly_features', PolynomialFeatures(degree=5, include_bias=False)),
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

# Prep data with pipeline

# Pipeline takes an excessively long time to transform data
# Has not been able to complete on my machine
X_poly = X_poly_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_poly = X_poly_pipeline.transform(X_train)
X_test_poly = X_poly_pipeline.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

preds = lin_reg.predict(X_train_poly)
x.append(get_model_stats('polynomial_regression', y_train, preds))

# bayes

X_bayes_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', poly_pipeline, num_df_list),
])

X_bayes = X_bayes_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_bayes = X_bayes_pipeline.transform(X_train)
X_test_bayes = X_bayes_pipeline.transform(X_test)
baesianReg = linear_model.BayesianRidge()
baesianReg.fit(X_train_bayes.todense(),y_train)
preds = baesianReg.predict(X_train_bayes)
x.append(get_model_stats('bayesian_regression', y_train, preds))





X_gb_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_df_list),
    ('num', poly_pipeline, num_df_list),
])

X_gb = X_gb_pipeline.fit_transform(X)  # Whole set ran through pipeline for cross-val
X_train_gb = X_gb_pipeline.transform(X_train)
X_test_gb = X_gb_pipeline.transform(X_test)
model = GradientBoostingRegressor(n_estimators=400,random_state=21)
kfold = KFold(n_splits=10, random_state=21)

model.fit(X_train_gb, y_train)
preds =model.predict(X_train_gb)
x.append(get_model_stats('grad_boost', y_train, preds))


loss = ['ls', 'lad', 'huber']
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
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

model = GradientBoostingRegressor(n_estimators=400,random_state=21)
rs = RandomizedSearchCV(estimator=model,
            param_distributions=hyperparameter_grid,
            cv=4, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5,
            return_train_score = True,
            random_state=42)
rs.fit(X_train_gb, y_train)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))