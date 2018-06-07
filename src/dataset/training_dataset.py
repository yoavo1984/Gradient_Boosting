from src.dataset.dataset import Dataset


class TrainingDataset(Dataset):
    def __init__(self, data_subset):
        super().__init__(data_subset)

    def code_categorial_features(self):
        """
        For each possible value (including None) of feature i compute the average of target.
        Randk the values according to the average compute value(ascending).
        * Saves the coding map for use on the test set.
        :return: 
        """
        pass

    def mean_imputation(self):
        """
        For each numerical feature compute the average value over the training data excluding missing values, 
        We'll use this value to fill in for missing values.
        * Saves the imputation map for use on the test set.
        :return: 
        """
        pass