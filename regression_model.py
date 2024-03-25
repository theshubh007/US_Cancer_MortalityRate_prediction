import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stat
from data_processing import split_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm


def correlation_among_numberic_features(df, cols):
    numeric_col = df[cols]
    correlation = numeric_col.corr()
    correlation_feature = set()
    for i in range(len(correlation.columns)):
        for j in range(i):
            if abs(correlation.iloc[i, j]) > 0.8:
                colname = correlation.columns[i]
                correlation_feature.add(colname)
    return correlation_feature


def lr_model(x_train, y_train):
    lr_model = LinearRegression().fit(x_train, y_train)
    return lr_model


def Mean_Squared_Error(y_test, y_predict):
    L_y = np.linalg.norm(y_test - y_predict) ** 2
    return L_y


def rsquare(y_test, y_predict):
    return r2_score(y_test, y_predict)


def adjusted_rsquare(y_test, y_predict, p):
    r2 = r2_score(y_test, y_predict)
    n = len(y_test)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2


def mean_5_fold_cross_validation(model, x_train, y_train):
    mse = cross_val_score(model, x_train, y_train, scoring="r2", cv=5)
    return mse
