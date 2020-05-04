
from ..alloc import Allocation

class SlurmAllocation(Allocation):

    def __init__(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        super().__init__(nodes=nodes, ppn=ppn, duration=duration, **kwargs)

    def __str__():
        pass

    def __repr__():
        pass

    def get_alloc_cmd(self):
        """Return the command to request an allocation from Slurm with
           the class variables as the slurm options."""

        salloc = ["salloc",
                  "--no-shell",
                  "-N", str(self.nodes),
                  "--ntasks-per-node", str(self.ppn),
                  "--time", self.duration,
                  "-J", "SmartSim"]

        for opt, val in self.add_opts.items():
            prefix = "-" if len(str(opt)) == 1 else "--"
            if not val:
                salloc += [prefix + opt]
            else:
                salloc += ["=".join((prefix+opt, str(val)))]

        return salloc