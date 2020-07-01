import pandas as pd
import numpy as np
import os

def load_data_from_cvs_file(file_name):
    path_dir = os.getcwd()
    path_file = os.path.join(path_dir, 'Data', file_name + ".csv")
    if os.path.exists(path_file):
        dataset = pd.read_csv(path_file)
        return dataset
    else:
        print("Unable to find the file at {file_path}")


def minmax_column_in_dataset(dataset):
    """
    This function calculate the mininum value and maximum value in columns dataset
    :param dataset:
    :param columns:
    :return:
    """
    minmax = list()
    columns_dataset = dataset.columns[1:-1]    # omit order and label column in dataset
    for column in columns_dataset:
        print('columns   ', column)
        max_value = dataset[column].max()
        min_value = dataset[column].min()
        minmax.append([min_value, max_value])
    return minmax

def scaler(scale_value, minmax):
    scale_value = (scale_value - minmax[0])/(minmax[1] -minmax[0])
    return scale_value

def normalize_dataset(dataset, minmax_columns):
    columns_dataset = dataset.columns[1:-1]
    for column in columns_dataset:
        dataset[column].apply() #todo hihuhihiuhiius
    print('num column: ',len(dataset.columns[1:-1]))






df = load_data_from_cvs_file('data_train_model')
minmax_columns = minmax_column_in_dataset(df)
print(minmax_columns)  #minmax_columns[0][1]
print('value of first element', minmax_columns[0][1])
normalize_dataset(df, minmax_columns)