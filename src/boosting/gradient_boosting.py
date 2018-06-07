from src.tree.structure import RegressionTree, RegressionTreeEnsemble


def cart(max_depth, min_node_size):
    """
    Perform the CART decision tree algorithm
    :param max_depth: 
    :param min_node_size: 
    :return: 
    """
    return RegressionTree()


def gbrt(train_set, num_trees, max_depth, min_node_size ,test_set):
    """
    Preform the Gradient Boosted regression tree algorithm.
    :param num_trees: 
    :param max_depth: 
    :param min_node_size: 
    :return: 
    """
    tree_ensemble = RegressionTreeEnsemble()
    for _ in num_trees:
        new_tree = cart(max_depth, min_node_size)
        weight = 1
        tree_ensemble.add_tree(new_tree, weight)

    return tree_ensemble
