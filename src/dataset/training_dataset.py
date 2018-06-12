from src.dataset.dataset import Dataset
import pandas as pd


class TrainingDataset(Dataset):
    imputation_map = {}
    coding_map = {}

    def __init__(self, data_subset, target_name):
        super().__init__(data_subset, target_name)
        self.fill_missing_values()

    def code_categorial_features(self):
        """
        For each possible value (including None) of feature i compute the average of target.
        Randk the values according to the average compute value(ascending).
        * Saves the coding map for use on the test set.
        :return: 
        """
        obj_cols = self.dataframe.select_dtypes(include=['object']).columns

        coding_map = {}
        for col in obj_cols:
            unique_vals = self.dataframe[col].unique()
            unique_vals_dict = {}
            for val in unique_vals:
                if pd.isnull(val):
                    nan_mean = self.dataframe[self.dataframe[col].isnull()].y.mean()
                    unique_vals_dict[val] = nan_mean
                else:
                    val_mean = self.dataframe[self.dataframe[col] == val].y.mean()

                unique_vals_dict[val] = val_mean

            unique_vals_ranks_dict = {key: rank for rank, key in
                                      enumerate(sorted(unique_vals_dict,
                                                       key=unique_vals_dict.get, reverse=True), 1)}

            self.dataframe[col].replace(unique_vals_ranks_dict, inplace=True)
            coding_map[col] = unique_vals_ranks_dict

        self.coding_map = coding_map

    def mean_imputation(self):
        """
        For each numerical feature compute the average value over the training data excluding missing values, 
        We'll use this value to fill in for missing values.
        * Saves the imputation map for use on the test set.
        :return: 
        """
        imputation_map = self.dataframe.mean()
        self.dataframe.fillna(imputation_map, inplace=True)

        self.imputation_map = imputation_map

    def get_imputation_map(self):
        return self.imputation_map

    def get_coding_map(self):
        return self.coding_map

    def fill_missing_values(self):
        self.mean_imputation()
        self.code_categorial_features()

    def get_sample_mean(self, sample):
        return sample.y.mean()

    def get_training_data_with_y_columns(self):
        return self.dataframe.copy()



