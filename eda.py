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
sns.distplot(df['PRICE'], norm_hist=True, kde=False, bins=20, hist_kws={"alpha": 1}).set(xlabel='Sale Price',
                                                                                         ylabel='Count')
df = df[df["PRICE"] < 1200000]
sns.distplot(df['PRICE'], norm_hist=True, kde=False, bins=20, hist_kws={"alpha": 1}).set(xlabel='Sale Price',
                                                                                         ylabel='Count')

plt.show()

columns = pd.DataFrame({'column_names': df.columns, 'datatypes': df.dtypes})
columns.sort_values(by="datatypes")

numerical = columns[columns["datatypes"] != "object"]["column_names"].to_list()
categorical = columns[columns["datatypes"] == "object"]["column_names"].to_list()

numerical = ['BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'PRICE',
             'SALE_NUM', 'GBA',
             'BLDG_NUM', 'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA']
# 'ZIPCODE', 'LATITUDE', 'LONGITUDE'

categorical = ['HEAT', 'AC', 'QUALIFIED', 'STRUCT', 'SOURCE', 'WARD', 'QUADRANT', "ASSESSMENT_NBHD"]

for var in categorical:
    cp = sns.countplot(df[var])
    title = 'img/cat_var_' + var + '.png'
    for label in cp.get_xticklabels():
        label.set_rotation(45)
    cp.figure.savefig(title)
    cp.set_title(var)
    print("![" + var + "](./" + title + "){width=50%}")

for var in categorical:
    sns_plot = sns.boxplot(x=var, y='PRICE', data=df)
    sns_plot.set_title(var)
    title = 'img/cat_bivar_price_' + var + '.png'
    sns_plot.figure.savefig(title)
    for label in sns_plot.get_xticklabels():
        label.set_rotation(45)
    print("![" + var + "](./" + title + "){width=50%}")

sorted_nb = df.groupby(['ASSESSMENT_NBHD'])['PRICE'].median().sort_values()
zona = sns.boxplot(x=df['ASSESSMENT_NBHD'], y=df['PRICE'], order=list(sorted_nb.index))
for label in zona.get_xticklabels():
    label.set_rotation(45)

zona.figure.savefig("img/zona_median.png")


def remove_outlier(var):
    return df[np.abs(df[var] - df[var].mean()) < 5 * df[var].std()]


for var in numerical:
    plt.figure()
    title = 'img/numerical_hist_' + var + '.png'
    ax = df[var].hist()
    ax.set_title(var)
    plt.savefig(title)
    print("![" + var + "](./" + title + "){width=33%}")

    df = remove_outlier("FIREPLACES")
    df = remove_outlier("LANDAREA")
    #df = remove_outlier("KITCHENS")
   # df = remove_outlier("YR_RMDL")

df["LANDAREA"].hist();plt.show()


df.plot(
    kind="scatter",
    x="X",
    y="Y",
    figsize=(20,14),
    c="PRICE",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    alpha=0.4,
)
plt.show()
