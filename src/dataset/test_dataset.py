from src.dataset.dataset import Dataset


class TestDataset(Dataset):
    def __init__(self, data_subset, target_name,  coding_map, imputation_map):
        super().__init__(data_subset, target_name)
        self.fill_missing_values(coding_map, imputation_map)

    def code_categorial_features(self, coding_map):
        obj_cols = self.dataframe.select_dtypes(include=['object']).columns

        for col in obj_cols:
            self.dataframe[col].replace(coding_map[col], inplace=True)

    def mean_imputation(self, imputation_map):
        self.dataframe.fillna(imputation_map, inplace=True)

    def fill_missing_values(self, coding_map, imputation_map):
        self.mean_imputation(imputation_map)
        self.code_categorial_features(coding_map)