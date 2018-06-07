import numpy as np

# Local imports
from src.boosting.pratition import Partition
from src.tree.structure import RegressionTree, RegressionTreeEnsemble, RegressionTreeNode


def cart(training_set, max_depth, min_node_size):
    """
    Perform the CART decision tree algorithm
    :param max_depth: 
    :param min_node_size: 
    :return: 
    """

    # Build tree and tree root
    root = RegressionTreeNode()
    tree = RegressionTree(root)

    # Build a list of partition for each depth
    depth = [[] for _ in range(max_depth)]

    # Create the first partition
    root_partition = Partition(training_set, root)

    depth[0].append(root_partition)
    for k in range(max_depth-2):
        # Build tree
        for partition in depth[k]:
            node = partition.get_node()
            j, s = partition.get_optimal_partition(min_node_size)

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
    for partition in depth[max_depth-1]:
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

    return partition_left, partition_right

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

if __name__ == "__main__":
    pass