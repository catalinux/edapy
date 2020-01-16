import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from util import get_data

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
param_dist = {
              "subsamble": [7, 8, 9, 10, 11],
              "colsample_bytree ": [7, 8, 9, 10, 11],
              }

XGB_RS = RandomizedSearchCV(xgb.XGBRegressor(learning_rate=0.2,gamma=3,max_depth=10,n_estimators=100,min_child_weight=11), param_distributions=param_dist, n_iter=150, verbose=10, n_jobs=64)
XGB_RS.fit(X_train_rf, y_train)

print("\n========================================================")
print(" Results from Random Search ")
print("========================================================")
print("\n The best estimator across ALL searched params:\n", XGB_RS.best_estimator_)
print("\n The best score across ALL searched params:\n", XGB_RS.best_score_)
print("\n The best parameters across ALL searched params:\n", XGB_RS.best_params_)
print("\n ========================================================")

# best 1
#  {'n_estimators': 100, 'min_child_weight': 11, 'max_depth': 10, 'learning_rate': 0.2, 'gamma': 3}
