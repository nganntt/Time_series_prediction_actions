import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

path_dir = os.getcwd()


def cleaning_data(file_name):
    """
    This function processes missing data and conver timestamp to seconds.
    The processed data is saved in processed_data.csv file.
    :param data_file_name: name of csv file
    :return: dataframe
    """
    #path_dir = os.getcwd()
    # read data from csv file
    path_file_name = os.path.join(path_dir, 'Data',
                                  file_name + ".csv")
    dataset = pd.read_csv(path_file_name)

    # processing missing data
    dataset = dataset.drop('hydraulic_pump', axis=1)
    # delete NAN value, the beginning data of sensor should be delete
    dataset = dataset.dropna()
    #delete index number in data
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

    # conver timestamp
    def conver_datetime(timestamp):
        return pd.to_datetime(timestamp, unit='s')

    dataset['timestamp_conv'] = dataset['timestamp'].apply(conver_datetime)
    dataset = dataset.drop('timestamp', axis=1)

    # save data to csv
    path_file_csv = os.path.join(path_dir, 'Data', "processed_data.csv")
    dataset.to_csv(path_file_csv, index=False)
    print('Data after cleaning is save at : ', path_file_csv)
    return dataset


def label_rolling_window(file_name, wind_size):
    """
    The rolling_wind_for_label chooses label while rolling window.
    The most common label in a window is chosen to lable for the window
    :param dataset: dataframe
    :param wind_size: number of items in a window
    :return: array of lables
    """
    # load data file
    path_file_name = os.path.join(path_dir, 'Data',
                                  file_name + ".csv")
    dataset = pd.read_csv(path_file_name)

    # create numpy lable and assign label to numpy lable
    np_labels = np.empty(len(dataset), dtype=object)
    i = wind_size - 1
    while i < len(dataset):
        begin_wind = i - wind_size + 1 if i >= wind_size else 0
        labels_wind = dataset['activity'].iloc[begin_wind: i + 1]
        list_lables = labels_wind.values.tolist()

        # subfunction to find the most common label
        def most_common_label(lst_string):
            return max(set(lst_string), key=lst_string.count)

        # assgin label to numpy label
        label = most_common_label(list_lables)
        np_labels[i] = label
        i += 1
    return np_labels


def calculate_features(file_name, size_wind):
    """
    The function calculates the mean, variance and std of the features
    The calucation iterates on window sizes (9, 7, 5)
    :param dataframe: processed data without missing data
    :param size_wind: size of window to slide the calculation
    :return: dataframe
    """
    feature_list = ['engine_speed', 'hydraulic_drive_off', 'anchor',
                    'pvalve_drill_forward', 'bolt', 'boom_lift', 'boom_lower',
                    'boom_forward', 'boom_backward', 'drill_boom_turn_left',
                    'drill_boom_turn_right', 'drill_boom_turn_forward',
                    'drill_boom_turn_backward', 'beam_right', 'beam_left',
                    'drill_boom_in_anchor_position']

    # load data file
    path_file_name = os.path.join(path_dir, 'Data', file_name + ".csv")
    dataset = pd.read_csv(path_file_name, index_col='timestamp_conv')

    # calculate mean, var, std for each window sizes on the features
    for feature_name in feature_list:
        # name of the feature
        mean_feature_name = 'mean_' + feature_name
        var_feature_name = 'var_' + feature_name
        std_feature_name = 'std_' + feature_name

        # calculate the features
        dataset[mean_feature_name] = dataset[feature_name]. \
            rolling(window=size_wind).mean()
        dataset[var_feature_name] = dataset[feature_name]. \
            rolling(window=size_wind).var()
        dataset[std_feature_name] = dataset[feature_name]. \
            rolling(window=size_wind).std()
        # drop the current feature
        dataset = dataset.drop(feature_name, axis=1)
    return dataset


def calucaltion_and_concat_feature_and_label(file_name, wind_size):
    """
    The calc_feature_and_label calculates the features with mean, var, std and
    lable for data.
    :param file_name: file name of processed data
    :param wind_size: number of rows for each the calculation
    :return: dataframe
    """
    # calculate features
    dataset = calculate_features(file_name, wind_size)
    # select label for data
    array_lables = label_rolling_window(file_name, wind_size)
    # concat features and labels
    dataset['activity'] = array_lables
    # delete first rows of None value after sliding window on data
    dataset = dataset.iloc[wind_size:]
    # save data to csv file
    name_csv_file = 'wind_size_' + str(wind_size) + '_feature_extract.csv'
    path_file_csv = os.path.join(path_dir, 'Data', name_csv_file)
    dataset.to_csv(path_file_csv)
    print('Data of feature calulation is saved in : ', path_file_csv)
    return dataset


def select_features_for_model(file_name):
    """
    The select feature function chooses the useful feature from exploring data
     on plots. Useful feature have well distribution and having characteristics
     that separate the classes.
     The data is save in 'data_train_model'
    :param file_name: name of csv file having many creating features
    :return: dataframe with useful features
    """
    # load data file containing all creating features
    path_file_name = os.path.join(path_dir, 'Data',
                                  file_name + ".csv")
    dataset = pd.read_csv(path_file_name)
    features = [
        'mean_engine_speed', 'mean_hydraulic_drive_off',
        'mean_drill_boom_in_anchor_position', 'mean_pvalve_drill_forward',
        'mean_bolt', 'mean_boom_lift', 'mean_boom_lower', 'mean_boom_forward',
        'mean_boom_backward', 'mean_drill_boom_turn_left',
        'mean_drill_boom_turn_right', 'mean_drill_boom_turn_forward',
        'mean_drill_boom_turn_backward', 'mean_beam_right',
        'mean_beam_left', 'mean_anchor', 'activity'
    ]
    # take only useful feature from data
    dataset_useful_feature = dataset[features]
    # save data to csv file
    name_csv_file = 'data_train_model.csv'
    path_file_csv = os.path.join(path_dir, 'Data', name_csv_file)
    dataset_useful_feature.to_csv(path_file_csv, index=False)
    print('Data to train model is saved in: {path_file_csv} ')
    return dataset_useful_feature


def process_feature_extraction_e2e(file_name):
    """
    This function execute from end to end of the preprocessing data and
    extracting feature.
    :param file_name: Name of draw data from sensors
    :return: NA
    """
    # process missing data and converting timestamp to seconds
    #Data is save in processed data
    cleaning_data(file_name)

    # create new features with mean, var, std calcualtion
    #Data is saved in wind_size_x_feature extraction
    wind_sizes = [5, 7, 9]
    for wind_size in wind_sizes:
        calucaltion_and_concat_feature_and_label('processed_data',wind_size)

    #select features for model window size 7 have a good seperate for classification,
    # Data is save in data_train_model.csv
    select_features_for_model('wind_size_7_feature_extract')



# VisualizeRoofbolter:
#    """
#    Visaulization show the plots of feature for the feature distributions and
#    identify the most useful features in terms of separating the seven activity
#    classes ('Transitional Delay', 'Hole Setup', 'Machine Off', 'Traveling',
#       'Idle', 'Drilling', 'Anchoring').
#   """




def load_data(file_name):
    # read file from csv file
    path_file_name = os.path.join(path_dir, 'Data',
                                  file_name + ".csv")
    dataset = pd.read_csv(path_file_name)
    # drop timestamp feature
    dataset = dataset.drop('timestamp_conv', axis=1)
    return dataset


def pair_plot_all_features(dataset, feature_compr):
    """
    The pair_plot_features plots feature1 with other features in a pair plot
    :param dataset: dataframe
    :param feature: name of feature need to be compared
    :return:NA
    """
    # show distribution of the feature
    sns.distplot(dataset[feature_compr], kde=False, bins=4)
    plt.show()

    features_list = dataset.columns[1:]
    for feature in features_list:
        sns.scatterplot(x=feature_compr, y=feature, data=dataset, hue='activity')
        plt.show()


def get_information_dataframe():
    """
    This function to get general information about data
    :return: NA
    """
    # read data from csv file
    dataset = load_data()
    # print first 10 rows in data
    print(dataset.head(10))
    # print info about data
    print('Information of data: ', dataset.info())
    # print shape data
    print('Shape of data: ', dataset.shape)





#step by step process data to generate data to train model
#process_feature_extraction_e2e('data_case_study')












