class GBRTHyperparameters(object):
    number_of_trees = 1
    tree_width = 1
    tree_depth = 1
    nu = 1
    num_threshold = 1
    sample_size = 1

    def __init__(self, num_of_tress, tree_width, tree_depth, nu, num_threshold, sample_size):
        self.number_of_trees = num_of_tress
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.nu = nu
        self.num_threshold = num_threshold
        self.sample_size = sample_size

    def get_cart_hyperparameters(self):
        return CARTHyperparameters(self.tree_width, self.tree_depth, self.num_threshold)


class CARTHyperparameters(object):
    tree_width = 0
    tree_depth = 0
    num_threshold = 0

    def __init__(self, tree_width, tree_depth, num_threshold):
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.num_threshold = num_threshold
