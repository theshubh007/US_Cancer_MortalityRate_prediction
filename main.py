from regression_model import (
    correlation_among_numberic_features,
    lr_model,
    # identify_significant_variables,
    Mean_Squared_Error,
    mean_5_fold_cross_validation,
    adjusted_rsquare,
)
import pandas as pd
import numpy as np
from data_processing import split_data
from data_processing_Test import process_data
from sklearn.metrics import r2_score

import statsmodels.api as sm

if __name__ == "__main__":
    print("stage:1")
    process_data("DataStore/cancer_reg.csv", "DataStore/cancer_reg_processed.csv")
    capped_Data = pd.read_csv("DataStore/cancer_reg_processed.csv")
    print(capped_Data.shape)

    print("stage:2")
    # corr_features = correlation_among_numberic_features(
    #     capped_Data, capped_Data.columns
    # )

    # print(corr_features)
    print("stage:2.1")
    # high_corr_features = corr_features
    high_corr_features = [
        "PctEmpPrivCoverage",
        "PctPrivateCoverageAlone",
        "MedianAgeFemale",
        "upper_bound",
        "avgDeathsPerYear",
        "lower_bound",
        "mid_point",
        "PctPublicCoverageAlone",
        "popEst2015",
        "PctBlack",
        "PctPrivateCoverage",
        "PctMarriedHouseholds",
    ]

    cols = [col for col in capped_Data.columns if col not in high_corr_features]
    len(cols)

    print("stage:3")
    x_train, x_test, y_train, y_test = split_data(capped_Data, "TARGET_deathRate")
    lr = lr_model(x_train, y_train)

    p = len(lr.coef_)
    y_predict = lr.predict(x_test)
    print("Mean_Squared_Error: ", Mean_Squared_Error(y_test, y_predict))
    print("rsquare1: ", r2_score(y_test, y_predict))
    print(
        "Adjusted_rsquare: ",
        adjusted_rsquare(y_test, y_predict, p),
    )
    print(
        "rsquare mean 5 fold cross validation ",
        mean_5_fold_cross_validation(lr, x_train, y_train),
    )

    print("stage:4 OLS model")
    print(x_train.shape)
    x_train = sm.add_constant(x_train)
    print(x_train.shape)
    lr2 = sm.OLS(y_train, x_train).fit()
    x_test = sm.add_constant(x_test)
    ypred = lr2.predict(x_test)
    print(r2_score(y_test, ypred))
