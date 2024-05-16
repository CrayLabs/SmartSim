from __future__ import annotations
from enum import Enum
import typing as t
import copy

from smartsim.log import get_logger 

from .translators.batch.pbs import QsubBatchArgTranslator
from .translators.batch.slurm import SlurmBatchArgTranslator
from .translators.batch.lsf import BsubBatchArgTranslator
from .translators import BatchArgTranslator 

IntegerArgument_1 = t.Dict[str, t.Optional[int]]
FloatArgument_1 = t.Dict[str, t.Optional[float]]
StringArgument_1 = t.Dict[str, t.Optional[str]]                                                                             

logger = get_logger(__name__)

class SupportedLaunchers(Enum):
    """ Launchers that are supported by
    SmartSim.
    """
    pbs = "qsub"
    lsf = "bsub"
    slurm = "sbatch"

class BatchSettings():
    def __init__(
        self,
        scheduler: str,
        scheduler_args: t.Optional[t.Dict[str, t.Union[str,int,float,None]]] = None,
        env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        **kwargs: t.Any,
    ) -> None:
        scheduler_to_translator = {
            'sbatch' : SlurmBatchArgTranslator(),
            'jsrun' : BsubBatchArgTranslator(),
            'qsub' : QsubBatchArgTranslator(),
        }
        if scheduler in scheduler_to_translator:
            self.scheduler = scheduler
        else:
            raise ValueError(f"'{scheduler}' is not a valid scheduler name.")

        # TODO check and preprocess env_vars
        self.env_vars = env_vars or {}

        # TODO check and preporcess launcher_args
        self.scheduler_args = scheduler_args or {}
        self.arg_translator = t.cast(BatchArgTranslator,scheduler_to_translator.get(self.scheduler))

    @property
    def scheduler_args(self) -> t.Dict[str, t.Optional[str]]:
        """Retrieve attached batch arguments

        :returns: attached batch arguments
        """
        return self._scheduler_args

    @scheduler_args.setter
    def scheduler_args(self, value: t.Dict[str, t.Optional[str]]) -> None:
        """Attach batch arguments

        :param value: dictionary of batch arguments
        """
        self._scheduler_args = copy.deepcopy(value) if value else {}

    def set_walltime(self, walltime: str) -> None:
        """Set the walltime of the job

        format = "HH:MM:SS"

        :param walltime: wall time
        """
        # TODO check for formatting here
        args = self.arg_translator.set_walltime(walltime)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_nodes(self, num_nodes: int) -> None:
        """Set the number of nodes for this batch job

        :param num_nodes: number of nodes
        """
        args = self.arg_translator.set_nodes(num_nodes)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_account(self, account: str) -> None:
        """Set the account for this batch job

        :param account: account id
        """
        args = self.arg_translator.set_account(account)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_partition(self, partition: str) -> None:
        """Set the partition for the batch job

        :param partition: partition name
        """
        args = self.arg_translator.set_partition(partition)
        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_queue(self, queue: str) -> None:
        """alias for set_partition

        Sets the partition for the slurm batch job

        :param queue: the partition to run the batch job on
        """
        args = self.arg_translator.set_queue(queue)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_cpus_per_task(self, cpus_per_task: int) -> None:
        """Set the number of cpus to use per task

        This sets ``--cpus-per-task``

        :param num_cpus: number of cpus to use per task
        """
        args = self.arg_translator.set_cpus_per_task(cpus_per_task)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_hostlist(self, host_list: t.Union[str, t.List[str]]) -> None:
        """Specify the hostlist for this job

        :param host_list: hosts to launch on
        :raises TypeError: if not str or list of str
        """
        args = self.arg_translator.set_hostlist(host_list)
        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_smts(self, smts: int) -> None:
        """Set SMTs

        This sets ``-alloc_flags``. If the user sets
        SMT explicitly through ``-alloc_flags``, then that
        takes precedence.

        :param smts: SMT (e.g on Summit: 1, 2, or 4)
        """
        args = self.arg_translator.set_smts(smts)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_project(self, project: str) -> None:
        """Set the project

        This sets ``-P``.

        :param time: project name
        """
        args = self.arg_translator.set_project(project)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def set_tasks(self, tasks: int) -> None:
        """Set the number of tasks for this job

        This sets ``-n``

        :param tasks: number of tasks
        """
        args = self.arg_translator.set_tasks(tasks)
        if args:
            for key, value in args.items():
                self.set(key, value)
    
    def set_ncpus(self, num_cpus: int) -> None:
        """Set the number of cpus obtained in each node.

        If a select argument is provided in
        ``QsubBatchSettings.resources``, then
        this value will be overridden

        :param num_cpus: number of cpus per node in select
        """
        args = self.arg_translator.set_ncpus(num_cpus)
        if args:
            for key, value in args.items():
                self.set(key, value)

    def format_batch_args(self) -> t.List[str]:
        """Get the formatted batch arguments for a preview
        """
        return self.arg_translator.format_batch_args(self.scheduler_args)

    def set(self, key: str, arg: t.Union[str,int,float,None]) -> None:
        # Store custom arguments in the launcher_args
        self.scheduler_args[key] = arg