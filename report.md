## Abstract

The residential property descriptions and address point information is current as of July 2018 and is provided by D.C. Geographic Information System

## Eda


At first look:

- there are some columns that can't be used directly : FULLADRESS, CENSUS_BLOCK, SQUARE, X, Y (the last two are the same with latitude and longitude) 
- Some fields need processing. Like sell date into year and zipcode into categorical
- few colummns have NA VALUEs


### Types of data

```
table(sapply(df, class))
```

| factor       | integer | numeric |
|--------------|---------|---------|
| 23           | 11      | 15      |


## Data processing

From 158957 properties, only 98216 had price so all the analysis was made starting with this subset of data


### Looking at the target feature

There are 4655 outlier values. (what should with do with them?). An option would be  removed them as they represent less than 5% of data




### Missing Values

We are looking to the columns that are missing most of the values


| Column             | Percentage |
|--------------------|------------|
| NUM_UNITS          | **41.048** |
| AYB                | **0.114**  |
| YR_RMDL            | **41.278** |
| STORIES            | 41.082     |
| SALEDATE           | **0.001**  |
| GBA                | 41.048     |
| STYLE              | 41.048     |
| STRUCT             | 41.048     |
| GRADE              | 41.048     |
| CNDTN              | 41.048     |
| EXTWALL            | 41.048     |
| ROOF               | 41.048     |
| INTWALL            | 41.048     |
| KITCHENS           | **41.049** |
| CMPLX_NUM          | 58.952     |
| LIVING_GBA         | 58.952     |
| CITY               | 41.385     |
| STATE              | 41.385     |
| NATIONALGRID       | 41.385     |
| ASSESSMENT_SUBNBHD | 20.623     |
| QUADRANT           | **0.103**  |



From the columns with missed values I will keep only the following columns:  

- AYB, SALEDATE, QUADRANT (they miss only few values and I can drop the rows)
- KITCHEN,NUM_UNITS - can be imputed
- YR_RMDL - can be calculated from AYB 
- LIVING_GBA - can be imputed from GBA
- GBA - can be imputed from LIVING_GBA






### What type of variation occurs within my variables?

First we look at the price column, the value that I try to predict


![price hist](./img/price_hist_with_outliers.png){width=33%} ![price hist](./img/price_hist_without_outlier.png){width=33%}   ![price hist](./img/hist_price_log.png){width=33%} 


** Numerical

![NUM_UNITS](./img/numerical_hist_NUM_UNITS.png){width=33%} ![ROOMS](./img/numerical_hist_ROOMS.png){width=33%} ![BEDRM](./img/numerical_hist_BEDRM.png){width=33%}
![AYB](./img/numerical_hist_AYB.png){width=33%} ![YR_RMDL](./img/numerical_hist_YR_RMDL.png){width=33%} ![EYB](./img/numerical_hist_EYB.png){width=33%}
![STORIES](./img/numerical_hist_STORIES.png){width=33%} ![PRICE](./img/numerical_hist_PRICE.png){width=33%} ![SALE_NUM](./img/numerical_hist_SALE_NUM.png){width=33%}
![GBA](./img/numerical_hist_GBA.png){width=33%} ![BLDG_NUM](./img/numerical_hist_BLDG_NUM.png){width=33%} ![KITCHENS](./img/numerical_hist_KITCHENS.png){width=33%}
![FIREPLACES](./img/numerical_hist_FIREPLACES.png){width=33%} ![USECODE](./img/numerical_hist_USECODE.png){width=33%} ![LANDAREA](./img/numerical_hist_LANDAREA.png){width=33%}


A couple of feature have large outliers

** Categorical 

![HEAT](./img/cat_var_HEAT.png){width=50%} ![AC](./img/cat_var_AC.png){width=50%} 
![QUALIFIED](./img/cat_var_QUALIFIED.png){width=50%} ![STRUCT](./img/cat_var_STRUCT.png){width=50%}
![SOURCE](./img/cat_var_SOURCE.png){width=50%} ![WARD](./img/cat_var_WARD.png){width=50%} 
![QUADRANT](./img/cat_var_QUADRANT.png){width=50%} ![ASSESSMENT_SUBNBHD](./img/cat_var_ASSESSMENT_NBHD.png){width=50%}


### What type of covariation occurs between my variables?


Varianta between columns can be seen in the figure below

![corr](./img/corr.png)

** With target predictor: PRICE ** 


** Continouse 

** Categorical **

Let's see how categorical columns coralate with PRICE column 


![HEAT](./img/cat_bivar_price_HEAT.png){width=50%} ![AC](./img/cat_bivar_price_AC.png){width=50%}
![QUALIFIED](./img/cat_bivar_price_QUALIFIED.png){width=50%} ![STRUCT](./img/cat_bivar_price_STRUCT.png){width=50%} 
![SOURCE](./img/cat_bivar_price_SOURCE.png){width=50%} ![WARD](./img/cat_bivar_price_WARD.png){width=50%} 
![QUADRANT](./img/cat_bivar_price_QUADRANT.png){width=50%} ![ASSESSMENT_NBHD](./img/cat_bivar_price_ASSESSMENT_NBHD.png){width=50%}

We can see that per neighbourhod price range differs so we we zoom it a little bit per median price

![Meian price per neighbourhood ](./img/zona_median.png)


### Some intesting plot: plot Map

[plot on map](./img/imgplot.png)


Observation: we can see that North-West has expensive houses



## Regression Models

### Linear Regression

A simple regression is applied 


### Polynomial regression 

Polynomial Regression is a form of linear regression in which the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x)

### GradientBoost

| name                  | EV   | MAE       | r2   | RMSE      | max_error  |
|-----------------------|------|-----------|------|-----------|------------|
| linear_regression     | 0.63 | 116015.26 | 0.63 | 154794.57 | 2993115.91 |
| polynomial_regression | 0.63 | 116015.26 | 0.63 | 154794.57 | 2993115.91 |
| bayesian_regression   | 0.63 | 116039.17 | 0.63 | 154806.31 | 2986934.94 |