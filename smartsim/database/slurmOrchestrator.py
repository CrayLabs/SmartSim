from ..settings import SrunSettings, SbatchSettings
from .orchestrator import Orchestrator
from ..error import SmartSimError

class SlurmOrchestrator(Orchestrator):

    def __init__(self, port, db_nodes=1, batch=True, alloc=None, dpn=1, **kwargs):
        """Initialize an Orchestrator reference for Slurm based systems

        The orchestrator launches as a batch by default. The Slurm orchestrator
        can also be given an allocation to run on. If no allocation is provided,
        and batch=False, at launch, the orchestrator will look for an interactive
        allocation to launch on.

        :param port: TCP/IP port
        :type port: int
        :param db_nodes: number of database shards, defaults to 1
        :type db_nodes: int, optional
        :param batch: Run as a batch workload, defaults to True
        :type batch: bool, optional
        :param alloc: allocation to launch on, defaults to None
        :type alloc: str, optional
        :param dpn: number of database per node (MPMD), defaults to 1
        :type dpn: int, optional
        """
        super().__init__(port,
                         db_nodes=db_nodes,
                         batch=batch,
                         alloc=alloc,
                         dpn=dpn,
                         **kwargs)
        self.batch_settings = self._build_batch_settings(db_nodes, alloc, batch, **kwargs)

    def set_cpus(self, num_cpus):
        """Set the number of CPUs available to each database shard

        This effectively will determine how many cpus can be used for
        compute threads, background threads, and network I/O.

        :param num_cpus: number of cpus to set
        :type num_cpus: int
        """
        if self.batch:
            self.batch_settings.batch_args["cpus-per-task"] = num_cpus
        for db in self:
            db.run_settings.set_cpus_per_task(num_cpus)

    def set_walltime(self, walltime):
        """Set the batch walltime of the orchestrator

        Note: This will only effect orchestrators launched as a batch

        :param walltime: amount of time e.g. 10 hours is 10:00:00
        :type walltime: str
        :raises SmartSimError: if orchestrator isn't launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set walltime")
        self.batch_settings.set_walltime(walltime)

    def set_batch_arg(self, arg, value):
        """Set a Sbatch argument the orchestrator should launch with

        Some commonly used arguments such as --job-name are used
        by SmartSim and will not be allowed to be set.

        :param arg: batch argument to set e.g. "exclusive"
        :type arg: str
        :param value: batch param - set to None if no param value
        :type value: str | None
        :raises SmartSimError: if orchestrator not launching as batch
        """
        if not self.batch:
            raise SmartSimError("Not running as batch, cannot set batch_arg")
        # TODO catch commonly used arguments we use for SmartSim here
        self.batch_settings.batch_args[arg] = value

    def _build_run_settings(self, exe, exe_args, **kwargs):
        alloc = kwargs.get("alloc", None)
        dpn = kwargs.get("dpn", 1)
        run_args = kwargs.get("run_args", {})
        batch = kwargs.get("batch", True)

        # if user specified batch=False
        # also handles batch=False and alloc=False (alloc will be found by launcher)
        if alloc or not batch:
            run_args["nodes"] = 1
            run_args["ntasks"] = dpn
            run_args["ntasks-per-node"] = dpn
            run_settings = SrunSettings(exe, exe_args, run_args=run_args, alloc=alloc)
            if dpn > 1:
                # tell step to create a mpmd executable
                run_settings.mpmd = True
            return run_settings
        # if batched and user did not specify allocation
        else:
            srun_args = {"nodes": 1, "ntasks": dpn}
            run_settings = SrunSettings(exe, exe_args, run_args=srun_args)
            if dpn > 1:
                run_settings.mpmd = True
            return run_settings

    def _build_batch_settings(self, db_nodes, alloc, batch, **kwargs):
        batch_settings = None
        dpn = kwargs.get("dpn", 1)
        # enter this conditional if user has specified an allocation to run
        # on or if user specified batch=False (alloc will be found through env)
        if not alloc and batch:
            batch_args = {"nodes": db_nodes, "ntasks-per-node": dpn}
            batch_settings = SbatchSettings(batch_args=batch_args)
        return batch_settings
