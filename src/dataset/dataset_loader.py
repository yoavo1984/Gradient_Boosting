import pandas as pd
import numpy as np
import os

from src.dataset.test_dataset import TestDataset
from src.dataset.training_dataset import TrainingDataset


def read_data(path_to_file, data_name):
    data_path = os.path.join(path_to_file, "{}.csv".format(data_name))
    # Read the data file.
    df_all = pd.read_csv(data_path)

    # Remove the id field.
    if 'Id' in df_all:
        df_all.drop(['Id'], axis=1, inplace=True)

    # Remove entries for which the Saleprice attribute is not known.
    if 'SalePrice' in df_all:
        df_all = df_all[np.isfinite(df_all['SalePrice'])]

    return df_all


def split_data(df, threshold):
    df['train'] = np.random.rand(len(df)) < threshold

    train = df[df.train == 1]
    train.drop(['train'], axis=1, inplace=True)
    test = df[df.train == 0]
    test.drop(['train'], axis=1, inplace=True)

    split_data_dict = {'train': train, 'test': test}

    return split_data_dict


def create_raw_training_and_test_sets(dataframe, threshold):
    split_data_dict = split_data(dataframe, threshold)

    return split_data_dict["train"], split_data_dict["test"]


def create_data(path_to_file, data_name):
    threshold = 0.8
    dataframe = read_data(path_to_file, data_name)
    train, test = create_raw_training_and_test_sets(dataframe, threshold)
    train = TrainingDataset(train, "SalePrice")
    test = TestDataset(test, "SalePrice", train.get_coding_map(), train.get_imputation_map())
    return train, test


def create_real_data(path_to_file, data_name, test_name):
    train = read_data(path_to_file, data_name)
    test = read_data(path_to_file, test_name)
    train_set = TrainingDataset(train, "SalePrice")
    test_set = TestDataset(test, None, train_set.get_coding_map(), train_set.get_imputation_map())
    return train_set, test_set
