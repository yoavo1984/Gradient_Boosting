class RegressionTreeNode(object):
    # Index of the feature on which we split at this node.
    j = -1

    # The threshold on which we split this node
    s = -1

    # immediate left descent of the current node
    left_descendant = None

    # immediate right descent of the current node
    right_descendant = None

    # value of the node.
    const = 0

    def __init__(self):
        pass

    def make_terminal(self):
        """
        make the node into a terminal (leaf node) and set its constant value to c.
        :return: 
        """
        self.c = 10

    def split(self, j, s):
        # Set j and s
        self.j = j
        self.s = s

        # Instantiate left and right descendants.

    def print_sub_tree(self):
        pass


class RegressionTree(object):
    root = None

    def __init__(self):
        pass

    def get_root(self):
        return self.root

    def evaluate (x):
        """
        For a vector valued x compute the value of the function represented by the tree.
        :return: The predicted value of x by the tree.
        """
        return 0.0


class RegressionTreeEnsemble(object):
    # An ordered collection of type RegressionTree
    trees = []

    # The weight associated with each regression tree.
    weights = []

    # M - the number of regression trees.
    M = 0

    # c - the initial constant value returned before any tree are added.
    const = 0

    def __init__(self):
        pass

    def add_tree(self, tree, weight):
        self.trees.append(tree)
        self.weights.append(weight)

    def set_initial_constant(self, c):
        self.const = c

    def evaluate(self, x, m):
        evaluation_sum = 0
        for i in range(m):
            tree_evaluation = self.trees[i].evalute(x)
            tree_addition = self.weights[i] * tree_evaluation
            evaluation_sum += tree_addition

        return evaluation_sum

if __name__ == "__main__":
    rtn = RegressionTreeNode()
    print (rtn.get_a())