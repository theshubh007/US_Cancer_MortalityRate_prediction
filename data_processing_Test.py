from data_ingest import DataIngest
from data_processing import (find_constant_columns,
                             delete_constant_columns,
                             find_columns_with_few_values, drop_and_fill)
from feature_engineering import bin_to_num, one_hot_encoding


def process_data(input_file_path, output_file_path):
    data_ingest = DataIngest()
    df = data_ingest.load_data(input_file_path)

    constant_columns = find_constant_columns(df)
    print("const columns which contains single value: ", constant_columns)
    column_with_few_values = find_columns_with_few_values(df, 10)

    ##To handle column with Intervels/object type value
    type(df["binnedInc"][0])
    df = bin_to_num(df)

    df = one_hot_encoding(df)
    df = drop_and_fill(df)
    print(df.shape)
    df.to_csv(output_file_path, index=False)
# data_ingest=DataIngest()
# df=data_ingest.load_data('DataStore/cancer_reg.csv')


# constant_columns= find_constant_columns(df)
# print("const columns which contains single value: ",constant_columns)
# column_with_few_values=find_columns_with_few_values(df, 10) 

# ##To handle column with Intervels/object type value
# type(df["binnedInc"][0])
# df=bin_to_num(df)

# df= one_hot_encoding(df)
# df= drop_and_fill(df)
# print(df.shape)
# df.to_csv('DataStore/cancer_reg_processed.csv', index=False)
