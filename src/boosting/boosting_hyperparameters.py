class GBRTHyperparameters(object):
    num_trees = 1
    min_node_size = 1
    max_depth = 1
    shrinkage = 1
    num_threshold = 1
    sample_size = 1

    def __init__(self, num_trees, min_node_size, max_depth, shrinkage, num_threshold, sample_size):
        self.num_trees = num_trees
        self.tree_width = min_node_size
        self.max_depth = max_depth
        self.shrinkage = shrinkage
        self.num_threshold = num_threshold
        self.sample_size = sample_size

    def get_cart_hyperparameters(self):
        return CARTHyperparameters(self.tree_width, self.max_depth, self.num_threshold)


class CARTHyperparameters(object):
    min_node_size = 0
    max_depth = 0
    num_threshold = 0

    def __init__(self, min_node_size, max_depth, num_threshold):
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.num_threshold = num_threshold
