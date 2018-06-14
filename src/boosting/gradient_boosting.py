import numpy as np
import pandas as pd

# Local imports
from src.boosting.pratition import Partition
from src.tree.structure import RegressionTree, RegressionTreeEnsemble, RegressionTreeNode


def cart(training_set, hyperparameters):
    """
    Perform the CART decision tree algorithm
    :param training_set: 
    :param max_depth: 
    :param min_node_size:
    :param num_thresholds:
    :return: 
    """

    # Build tree and tree root
    root = RegressionTreeNode()
    tree = RegressionTree(root)

    # Build a list of partition for each depth
    depth = [[] for _ in range(hyperparameters.max_depth)]

    # Create the first partition
    root_partition = Partition(training_set, root)

    depth[0].append(root_partition)
    for k in range(hyperparameters.max_depth-1):
        # Build tree
        for partition in depth[k]:
            node = partition.get_node()
            j, s = partition.get_optimal_partition(hyperparameters.min_node_size, hyperparameters.num_threshold)

            # Partition was found
            if j != -1:
                partition_left, partition_right = create_left_and_right_partition(node, partition, j, s)


                # Add nodes and partitions to the next depth
                depth[k+1].append(partition_left)
                depth[k+1].append(partition_right)

            # No partition found
            else:
                const = partition.get_partition_average()
                node.make_terminal(const)

    # Go over last level in the tree and make all nodes leaves.
    for partition in depth[hyperparameters.max_depth-1]:
        node = partition.get_node()
        const = partition.get_partition_average()
        node.make_terminal(const)

    return tree


def create_left_and_right_partition(node, partition, j, s):
    # Create nodes and partition
    node_left, node_right = node.split(j, s)
    partition_left, partition_right = partition.split(j, s)
    partition_left.set_node(node_left)
    partition_right.set_node(node_right)

    # updating node error
    node_left.error = partition_left.get_error()
    node_right.error = partition_right.get_error()

    return partition_left, partition_right


def gbrt(train_set, hyperparameters, test_set=None):
    """
    Preform the Gradient Boosted regression tree algorithm.
    :param train_set: 
    :param hyperparameters:
    :param test_set:
    :return: 
    """
    # Create the ensemble object
    tree_ensemble = RegressionTreeEnsemble()

    # Compute f_0
    f0 = train_set.get_dataset_target_mean()
    tree_ensemble.set_initial_constant(f0)

    for tree_number in range(hyperparameters.num_trees):
        # Get mini batch from training set
        instances = train_set.sample_minibatch(hyperparameters.sampling_portion)

        # Compute residual for each instance in mini-batch
        instances["y"] = instances.apply(lambda row: -1 * (row['y'] - tree_ensemble.evaluate(row, tree_number)), axis=1)

        # Build new tree using CART
        new_tree = cart(instances, hyperparameters.get_cart_hyperparameters())

        # Compute new tree weight
        nominator = instances.apply(lambda row: row["y"]*(new_tree.evaluate(row)), axis=1).sum()
        denominator = instances.apply(lambda row: (new_tree.evaluate(row))**2, axis=1).sum()
        weight = nominator / denominator * hyperparameters.shrinkage

        # Add tree to ensemble
        tree_ensemble.add_tree(new_tree, weight)
        print("New Tree weight = {}".format(weight))

        # Compute training error
        if test_set:
            print_train_test_error(train_set, test_set, tree_ensemble, tree_number)

    return tree_ensemble


def print_train_test_error(train_set, test_set, tree_ensemble, tree_number):
    train_instances = train_set.get_dataframe_copy()
    test_instances = test_set.get_dataframe_copy()

    train_rmse = tree_ensemble.compute_dataset_rmse(train_instances, tree_number+1)
    test_rmse = tree_ensemble.compute_dataset_rmse(test_instances, tree_number+1)

    print("Train cost after {} trees is : {}".format(tree_number + 1, train_rmse))
    print("Test cost after {} trees is : {}".format(tree_number + 1, test_rmse))

    with open("error.txt", "a") as myfile:
        myfile.write("{}. Train rmse = {}| Test rmse = {}".format(tree_number, train_rmse, test_rmse))

if __name__ == "__main__":
    np.random.seed(125)

    data = np.random.randint(0, 20, size=(10,3))
    df = pd.DataFrame(data, columns=["x1", "x2", "y"])
    tree = cart(df, 3, 2)
    print(tree)