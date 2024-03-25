from sklearn.model_selection import train_test_split


def find_constant_columns(df):
    constant_columns = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            constant_columns.append(column)
    return constant_columns


def delete_constant_columns(df, constant_columns):
    df.drop(constant_columns, axis=1, inplace=True)
    return df


def find_columns_with_few_values(df, threshold):
    few_value_columns = []
    for column in df.columns:
        if len(df[column].unique()) < threshold:
            few_value_columns.append(column)
    return few_value_columns


def find_duplicate_columns(df):
    duplicate_rows = df[df.duplicated()]
    return duplicate_rows


def delete_duplicate_columns(df):
    df.drop_duplicates(keep="first")
    return df


def drop_and_fill(df):
    cols_to_drop = df.columns[df.isnull().mean() > 0.5]
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=42
    )
    return X_train, X_test, y_train, y_test
