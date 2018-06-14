class GBRTHyperparameters(object):
    num_trees = 1
    min_node_size = 1
    max_depth = 1
    shrinkage = 1
    num_threshold = 1
    sampling_portion = 1

    def __init__(self, num_trees, min_node_size, max_depth, shrinkage, sampling_portion, num_threshold):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.shrinkage = shrinkage
        self.num_threshold = num_threshold
        self.sampling_portion = sampling_portion

    def get_cart_hyperparameters(self):
        return CARTHyperparameters(self.max_depth, self.min_node_size, self.num_threshold)


class CARTHyperparameters(object):
    min_node_size = 0
    max_depth = 0
    num_threshold = 0

    def __init__(self, max_depth, min_node_size, num_threshold):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.num_threshold = num_threshold
