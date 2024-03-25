import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def bin_to_num(df):
    binneInc = []
    for i in df["binnedInc"]:
        i = i.strip("()[]")  # remove brackets
        i = i.split(", ")  # split string into list
        i = tuple(map(float, i))  # convert to float type
        i = list(i)
        binneInc.append(i)
    df["binnedInc"] = binneInc
    df["lower_bound"] = [i[0] for i in df["binnedInc"]]
    df["upper_bound"] = [i[1] for i in df["binnedInc"]]
    df["mid_point"] = (df["lower_bound"] + df["upper_bound"]) / 2
    df.drop("binnedInc", axis=1, inplace=True)
    return df


def cat_to_col(df):
    df["county"] = [i.split(",")[0] for i in df["Geography"]]
    df["state"] = [i.split(",")[1] for i in df["Geography"]]
    df.drop("Geography", axis=1, inplace=True)
    return df


def one_hot_encoding(X):
    categorical_columns = X.select_dtypes(include=["object"]).columns
    # one hot encode categorical columns
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    hot_encoded = one_hot_encoder.fit_transform(X[categorical_columns])
    # convert the encoded sparse matrix into a dense array
    hot_encoded_dense = hot_encoded.toarray()
    # convert the encoded array into a dataframe
    hot_encoded_df = pd.DataFrame(
        hot_encoded_dense,
        columns=one_hot_encoder.get_feature_names_out(categorical_columns),
    )
    # drop the original categorical columns from the dataframe
    X = X.drop(categorical_columns, axis=1)
    # concatenate the original dataframe with the encoded dataframe
    X = pd.concat([X, hot_encoded_df], axis=1)
    return X
