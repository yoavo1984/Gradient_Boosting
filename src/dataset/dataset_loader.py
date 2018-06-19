import pandas as pd
import numpy as np
import os
from shutil import copyfile
# from sklearn.cross_validation import train_test_split

from src.dataset.test_dataset import TestDataset
from src.dataset.training_dataset import TrainingDataset


def read_data(path_to_file):
    DATA_PATH = os.path.join(path_to_file, 'songs.csv')
    COPY_PATH = os.path.join(path_to_file, 'songs_Copy.csv')
    # Read the data file.
    df_all = pd.read_csv(DATA_PATH)
    # save a copy of the data
    copyfile(DATA_PATH, COPY_PATH)
    # Remove the id field.
    # df_all.drop(['Id'], axis=1, inplace=True)
    # Remove entries for which the Saleprice attribute is not known.
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
    # if can use sklearn split
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(df_all.drop('SalePrice', axis=1),
    #                      df_all.SalePrice, test_size=0.2, random_state=42)

    split_data_dict = split_data(dataframe, threshold)

    return split_data_dict["train"], split_data_dict["test"]


def create_data(path_to_file):
    threshold = 0.8
    dataframe = read_data(path_to_file)
    train, test = create_raw_training_and_test_sets(dataframe, threshold)
    train = TrainingDataset(train, "SalePrice")
    test = TestDataset(test, "SalePrice", train.get_coding_map(), train.get_imputation_map())
    return train, test

