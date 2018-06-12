class Dataset(object):
    dataframe = None

    def __init__(self, data_subset, target_name):
        # Rename target value to y for consistency
        data_subset = data_subset.rename(index=str, columns={target_name: "y"})

        # Save dataframe
        self.dataframe = data_subset

    def get_dataset_target_mean(self):
        return self.dataframe.y.mean()

    def get_dataframe_copy(self):
        return self.dataframe.copy()

    def sample_minibatch(self, p=1):
        batch = self.dataframe.sample(frac=p, replace=False, weights=None).copy()
        return batch

    def get_x(self):
        X, y = split_data['train'].drop('SalePrice', axis=1), split_data['train'].SalePrice
        return X

    def get_y(self):
        X, y = split_data['train'].drop('SalePrice', axis=1), split_data['train'].SalePrice
        return y
