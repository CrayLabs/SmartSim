

class AllocManager:
    """The AllocManager holds all the information regarding allocations
       added or obtained by the user. The class is a wrapper around a
       dictionary that provides some convenience methods for interfacing
       with workload manager obtained allocations stored in an
       Allocation instance. Depending on the workload manager, different
       child implementations of the Allocation class may be used.
    """
    def __init__(self):
        """Initialize an AllocManager
        """
        self.allocs = {}  # alloc_id: Allocation

    def __call__(self):
        return self.allocs

    def __getitem__(self, alloc_id):
        return self.allocs[alloc_id]

    def add_step(self, alloc_id):
        """Add a job step

        This function increments the internal counter of
        job steps that have been launched on this allocation.

        :param alloc_id: allocation to increment steps of
        :type alloc_id: str
        :return: number of steps currently launched on the
                 allocation + 1
        :rtype: int
        """
        step = self.allocs[alloc_id].steps
        self.allocs[alloc_id].steps += 1
        return step

    def add(self, alloc_id, allocation):
        """Add an allocation the AllocManager

        :param alloc_id: The allocation id
        :type alloc_id: str
        :param allocation: The allocation object
        :type allocation: Allocation
        """
        self.allocs[alloc_id] = allocation

    def remove(self, alloc_id):
        """Remove an allocation from the AllocManager completely

        :param alloc_id: The allocation id
        :type alloc_id: str
        """
        del self.allocs[alloc_id]

    def verify(self, allocation):
        """Check that there exists an allocation that is sufficient

        :param allocation: The Allocation object
        :type allocation: Allocation
        """
        pass
