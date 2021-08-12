class ComputeNode:  # cov-slurm
    """The ComputeNode class holds resource information
    about a physical compute node
    """

    def __init__(self, node_name=None, node_ppn=None):
        """Initialize a ComputeNode

        :param node_name: the name of the node
        :type node_name: str
        :param node_ppn: the number of ppn
        :type node_ppn: int
        """
        self.name = node_name
        self.ppn = node_ppn

    def _is_valid_node(self):
        """Check if the node is complete

        Currently, validity is judged by name
        and ppn being not None.

        :returns: True if valid, false otherwise
        :rtype: bool
        """
        if self.name is None:
            return False
        if self.ppn is None:
            return False

        return True


class Partition:  # cov-slurm
    """The partition class holds information about
    a system partition.
    """

    def __init__(self):
        """Initialize a system partition"""
        self.name = None
        self.min_ppn = None
        self.nodes = set()

    def _is_valid_partition(self):
        """Check if the partition is valid

        Currently, validity is judged by name
        and each ComputeNode being valid

        :returns: True if valid, false otherwise
        :rtype: bool
        """
        if self.name is None:
            return False
        if len(self.nodes) <= 0:
            return False
        for node in self.nodes:
            if not node._is_valid_node():
                return False

        return True
