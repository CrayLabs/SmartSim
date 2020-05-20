

class AllocManager:
    """The AllocManager holds all the information regarding allocations
       added or obtained by the user. The class is a wrapper around a
       dictionary that provides some convience methods for interfacing
       with workload manager obtained allocations stored in an
       Allocation instance. Depending on the workload manager, different
       child implementations of the Allocation class may be used.
    """

    def __init__(self):
        self.allocs = {}  # alloc_id: Allocation

    def __call__(self):
        return self.allocs

    def __getitem__(self, alloc_id):
        return self.allocs[alloc_id]

    def add_step(self, alloc_id):
        """Increment the internal counter of job steps that
           have been launched on this allocation.

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
        self.allocs[alloc_id] = allocation

    def remove(self, alloc_id):
        """Remove an allocation from the AllocManager completely"""
        del self.allocs[alloc_id]

    def verify(self, allocation):
        """check that there exists an allocation that is sufficent for
           the requested workload
        """
        pass
