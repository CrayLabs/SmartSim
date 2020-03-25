

class Allocation:

    def __init__(self, alloc_id, partition, nodes):
        self.alloc_id = alloc_id
        self.partition = partition
        self.nodes = int(nodes)
        self.subjobs = 0

    def __str__(self):
        partition = self.partition
        if not partition:
            partition = "default"
        alloc_str = f"Allocation ID: {self.alloc_id} \n"
        alloc_str += f" - Partition: {partition} \n"
        alloc_str += f" - Nodes: {self.nodes} \n"
        return alloc_str


    def __repr__(self):
        partition = self.partition
        if not partition:
            partition = "default"
        return f"{self.alloc_id}-{partition}"

    def __ge__(self, other):
        if self.partition == other.partition:
            if self.nodes >= other.nodes:
                return True
        return False


class AllocManager:

    def __init__(self):
        self.allocs = {}  # alloc_id: Allocation

    def __call__(self):
        return self.allocs

    def __getitem__(self, partition):
        for alloc in self.allocs.values():
            if alloc.partition == partition:
                return alloc.alloc_id
        raise KeyError

    def __contains__(self, partition):
        for alloc in self.allocs.values():
            if alloc.partition == partition:
                return True
        return False

    def add_subjob(self, alloc_id):
        subjob_id = ".".join((alloc_id, str(self.allocs[alloc_id].subjobs)))
        self.allocs[alloc_id].subjobs += 1
        return subjob_id

    def add_alloc(self, alloc_id, partition, nodes):
        alloc = Allocation(alloc_id, partition, nodes)
        self.allocs[alloc_id] = alloc

    def remove_alloc(self, alloc_id):
        """Remove an allocation from the AllocManager completely"""
        del self.allocs[alloc_id]

    def verify(self, partition, nodes):
        """check that there exists an allocation that is sufficent for
           the requested workload
        """
        alloc_to_check = Allocation("not-allocated", partition, nodes)
        for alloc in self.allocs.values():
            if alloc >= alloc_to_check:
                return True
        return False
