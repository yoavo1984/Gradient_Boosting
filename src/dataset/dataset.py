class Dataset(object):
    dataframe = None

    def __init__(self, data_subset):
        self.dataframe = data_subset

    def get_x(self):
        X, y = split_data['train'].drop('SalePrice', axis=1), split_data['train'].SalePrice
        return X

    def get_y(self):
        X, y = split_data['train'].drop('SalePrice', axis=1), split_data['train'].SalePrice
        return y
