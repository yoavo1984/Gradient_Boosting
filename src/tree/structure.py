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
    const = -1

    # Flag for leaf or not
    leaf = False

    # node error
    error = 0

    def __init__(self):
        pass

    def make_terminal(self, c):
        """
        make the node into a terminal (leaf node) and set its constant value to c.
        :return: 
        """
        self.const = c
        self.leaf = True

    def split(self, j, s):
        # Set j and s
        self.j = j
        self.s = s

        # Instantiate left and right descendants.
        self.left_descendant = RegressionTreeNode()
        self.right_descendant = RegressionTreeNode()

        return self.left_descendant, self.right_descendant

    def print_sub_tree(self):
        pass

    def is_leaf(self):
        return self.leaf


class RegressionTree(object):
    root = None

    def __init__(self, root):
        self.root = root

    def get_root(self):
        return self.root

    def evaluate (self, x):
        """
        For a vector valued x compute the value of the function represented by the tree.
        :return: The predicted value of x by the tree.
        """
        # start with the root.
        current_node = self.root

        # While node is not a leaf.
        while not current_node.is_leaf():
            j = current_node.j
            s = current_node.s

            # split according to current node.
            if x[j] <= s:
                current_node = current_node.left_descendant
            else:
                current_node = current_node.right_descendant

        return current_node.const


class RegressionTreeEnsemble(object):
    # An ordered collection of type RegressionTree
    trees = []

    # The weights (beta) associated with each regression tree.
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
        self.M += 1

    def set_initial_constant(self, c):
        self.const = c

    def evaluate(self, x, m):
        evaluation_sum = self.const
        for i in range(m):
            tree_evaluation = self.trees[i].evaluate(x)

            tree_addition = self.weights[i] * tree_evaluation
            evaluation_sum -= tree_addition

        return evaluation_sum

    def __iter__(self):
        for tree in self.trees:
            yield tree

if __name__ == "__main__":
    rtn = RegressionTreeNode()
    print (rtn.get_a())