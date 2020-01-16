import numpy as np
import xgboost as xgb
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from util import get_data
from util import get_model_stats

df = get_data()
x = []

cat_df_list = list(df.select_dtypes(include=['object']))
num_df_list = list(df.select_dtypes(include=['float64', 'int64']))
num_df_list.remove('PRICE')

X = df.drop(columns=['PRICE'])
y = df['PRICE']
y_sqrt = np.sqrt(df['PRICE'])

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

XGB_RS = RandomizedSearchCV(xgb.XGBRegressor(), param_distributions=param_dist, n_iter=150, verbose=10, n_jobs= 14)
XGB_RS.fit(X_train_rf, y_train)
