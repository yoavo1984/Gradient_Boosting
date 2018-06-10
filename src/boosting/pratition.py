class Partition(object):
    # The instances in the partition.
    instances = None

    # The node associated with the partition.
    node = None

    def __init__(self, instances, node=None):
        self.instances = instances
        self.node = node

    def set_node(self, node):
        self.node = node

    def get_node(self):
        return self.node

    @staticmethod
    def cost(values, prediction):
        sum = 0

        for val in values:
            sum += pow(val - prediction, 2)

        return sum

    @staticmethod
    def compute_split_cost(left, right):
        true_left = left["y"].values
        true_right = right["y"].values

        left_cost = Partition.cost(true_left, left["y"].mean())
        right_cost = Partition.cost(true_right, right["y"].mean())

        return left_cost + right_cost

    def get_optimal_partition(self, min_node_size):
        instances = self.instances
        found_partition = False
        for feature in instances.columns[:-1]:
            values = instances[feature].unique()

            # We will save the best partition cost, feature and feature value.
            best_partition = (float("inf"), 0, 0)
            for val in values:
                left = instances[instances[feature] <= val]
                right = instances[instances[feature] > val]

                # We first check we have enough instances in each side.
                if left.shape[0] > min_node_size and right.shape[0] > min_node_size:
                    split_cost = self.compute_split_cost(left, right)
                    found_partition = True

                    if split_cost < best_partition[0]:
                        best_partition = (split_cost, feature, val)

        # If a partition was found return the best one.
        if found_partition:
            return best_partition[1], best_partition[2]
        # Return an accepted value to indicate no partition was found.
        else:
            return -1, -1


    def split(self, j, s):
        """
        splits the instnaces in the partition into 2 according to supplied j,s 
        :param j: The index of the feature
        :param s: The threshold for the split
        :return: Two instances arrays.
        """
        left  = self.instances[self.instances[j] <= s]
        right = self.instances[self.instances[j] > s]
        return Partition(left), Partition(right)

    def get_partition_average(self):
        """
        Compute the average value across all the instances in the partition
        :return: The average value across all the instances.
        """
        return self.instances["y"].mean()
