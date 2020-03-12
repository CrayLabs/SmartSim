
class AllocHandler:

    def __init__(self):
        self.partitions = {}  # partition : (node count: ppn)
        self.allocs = {} # partition : allocation_id

    def _add_to_allocs(self, run_settings):
        """Add an entities run_settings to an allocation
           Add up the total number of nodes or each partition and
           take the highest ppn value present in any of the run settings

           :param dict run_settings: dictionary of settings that include
                                     number of nodes, ppn and partition
                                     requested by the user.
        """
        # partition will be None if one is not listed
        part = run_settings["partition"]
        if not part:
            part = "default" # use default partition
        nodes = int(run_settings["nodes"])
        ppn = int(run_settings["ppn"])
        if part in self.partitions.keys():
            self.partitions[part][0] += nodes
            if self.partitions[part][1] < ppn:
                self.partitions[part][1] = ppn
        else:
            self.partitions[part] = [nodes, ppn]

    def _remove_alloc(self, partition):
        """Remove a partition from both the active allocations and the
           partitions dictionary. This is called when an allocation is
           released by the user.

           :param str partition: supplied by release. partition to be freed
        """
        # accept keyerror for the case where partition has been populated
        # but no allocation has been given
        try:
            self.partitions.pop(partition)
            self.allocs.pop(partition)
        except KeyError:
            pass