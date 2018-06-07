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

    def get_optimal_partition(self, min_node_size):
        return 1, 2

    def split(self, j, s):
        """
        splits the instnaces in the partition into 2 according to supplied j,s 
        :param j: The index of the feature
        :param s: The threshold for the split
        :return: Two instances arrays.
        """
        left, right = self.instances.split(j,s)
        return Partition(left), Partition(right)

    def get_partition_average(self):
        """
        Compute the average value across all the instances in the partition
        :return: The average value across all the instances.
        """
        return 0.0
