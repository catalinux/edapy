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
df = df.drop(['Unnamed: 0', 'GIS_LAST_MOD_DTTM', "FULLADDRESS", "CENSUS_BLOCK", "SQUARE", "X", "Y", "CENSUS_TRACT",
              "CENSUS_BLOCK"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# eda
df_missing = df.copy()
missing = df_missing.isnull().sum()
missing
np.round(100 * missing[missing > 0] / df.__len__(), 3).to_clipboard()

df = df.drop(
    ["SALEDATE", "STYLE", "GRADE", "CNDTN", "EXTWALL", "ROOF", "INTWALL", "CMPLX_NUM", "LIVING_GBA", "CITY", "STATE",
     "NATIONALGRID", "ASSESSMENT_SUBNBHD"], axis=1)

corr = df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

df = df[df["HEAT"] != 'No Data']

df = df[df["PRICE"] < 1200000]
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    df['PRICE'], norm_hist=True, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Sale Price', ylabel='Count');

plt.show()

columns = pd.DataFrame({'column_names': df.columns, 'datatypes': df.dtypes})
columns.sort_values(by="datatypes")

numerical = columns[columns["datatypes"] != "object"]["column_names"].to_list()
categorical = columns[columns["datatypes"] == "object"]["column_names"].to_list()

numerical = ['BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'PRICE',
             'SALE_NUM', 'GBA',
             'BLDG_NUM', 'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA', 'ZIPCODE', 'LATITUDE', 'LONGITUDE']

categorical = ['HEAT', 'AC', 'QUALIFIED', 'STRUCT', 'SOURCE', 'WARD', 'QUADRANT']

for var in categorical:
    cp = sns.countplot(df[var])
    title = 'img/cat_var_' + var + '.png'
    for label in cp.get_xticklabels():
        label.set_rotation(45)
    cp.figure.savefig(title)
    cp.set_xlabel(var, fontsize=14)
    print("![" + var + "](./" + title + "){width=33%}")

for var in categorical:
    sns_plot = sns.boxplot(x=var, y='PRICE', data=df)
    title = 'img/cat_var_price_' + var + '.png'
    for label in sns_plot.get_xticklabels():
        label.set_rotation(45)
    sns_plot.figure.savefig(title)
    sns_plot.set_xlabel(var, fontsize=14)
    print("![" + var + "](./" + title + "){width=33%}")

