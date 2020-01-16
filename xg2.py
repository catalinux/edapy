import numpy as np
import xgboost as xgb
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from util2 import get_data
from util2 import get_model_stats

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

m = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=3,
             importance_type='gain', learning_rate=0.2, max_delta_step=0,
             max_depth=10, min_child_weight=11, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

m.fit(X_train_rf, y_train)
y_pred_test = m.predict(X_test_rf)

x = get_model_stats('a', y_test, y_pred_test)
from scipy import stats

print(x)
print(df.shape)