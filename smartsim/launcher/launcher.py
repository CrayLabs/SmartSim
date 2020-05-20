import os
import abc
import time
import atexit
import zmq
import pickle
from subprocess import PIPE, Popen, CalledProcessError, TimeoutExpired, run

from .alloc import AllocManager
from .launcherUtil import seq_to_str
from ..error import LauncherError, SmartSimError
from .shell import execute_cmd, is_remote


from ..utils import get_logger, get_env
logger = get_logger(__name__)


class Launcher(abc.ABC):

    def __init__(self, *args, **kwargs):
        self.alloc_manager = AllocManager()
        super().__init__()

    #-------------- Abstract Methods --------------
    @abc.abstractmethod
    def validate(self, nodes=None, ppn=None, partition=None):
        """Validate the functionality of the launcher and availability of
           resources on the system

        :param nodes: Override the number of nodes to validate
        :param ppn: Override the processes per node to validate
        :param partition: Override the partition to validate
        :return:
        """
        pass

    @abc.abstractmethod
    def create_step(self, name, run_settings, multi_prog=False):
        """Convert a smartsim entity run_settings into a job step
           to be launched on an allocation. An entity must have an
           allocation assigned to it in the running settings or
           create_step will throw a LauncherError

        :param name: name of the step to be launch, usually entity.name
        :type name: str
        :param run_settings: smartsim run_settings for an entity
        :type run_settings: dict
        :param multi_prog: create a multi-program step, defaults to False
        :type multi_prog: bool, optional
        :raises LauncherError: if step creation fails
        :return: Step object
        """
        pass

    @abc.abstractmethod
    def get_step_status(self, step_id):
        """Return the status of a job step from either the OS or
           the workload manager.

        :param step_id: id of the step in the form of xxxxxx.x
        :type step_id: str
        :return: status of the job step and returncode
        :rtype: tuple of (str, str)
        """
        pass

    @abc.abstractmethod
    def get_step_nodes(self, step_id):
        """Return the compute nodes of a specific job or allocation
           in a list with the duplicates removed.

        :param job_id: job step id or allocation id
        :type job_id: str
        :raises LauncherError: if allocation or job step cannot be
                               found
        :return: list of compute nodes the job was launched on
        :rtype: list of str
        """
        pass

    @abc.abstractmethod
    def accept_alloc(self):
        """Accept a user provided and obtained allocation into the
           Launcher for future launching of entities. Obtain as much
           information about the allocation as possible by querying
           the workload manager.

        :param alloc_id: id of the allocation
        :type alloc_id: str
        :raises LauncherError: if the allocation cannot be found
        """
        pass

    @abc.abstractmethod
    def get_alloc(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        """Request an allocation with the specified arguments. Anything
           passed to the keywords args will be processed as a wlm
           argument and appended to the allocation command with the appropriate
           prefix (e.g. "-" or "--"). The requested allocation will be
           added to the AllocManager for launching entities.

           :param nodes: number of nodes for the allocation, defaults to 1
           :type nodes: int, optional
           :param ppn: number of tasks to run per node, defaults to 1
           :type ppn: int, optional
           :param duration: length of the allocation in HH:MM:SS format,
                           defaults to "1:00:00"
           :type duration: str, optional
           :raises LauncherError: if the allocation is not successful
           :return: the id of the allocation
           :rtype: str
        """
        pass

    @abc.abstractmethod
    def run(self, step):
        """Run a job step on an allocation through the workload manager
           A constructed job step is required such that the argument
           translation from SmartSimEntity to Step has been completed
           and an allocation has been assigned to the step.

        :param step: Step instance
        :type step: Step
        :raises LauncherError: If the allocation cannot be found or the
                               job step failed to launch.
        :return: job_step id
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def stop(self, step_id):
        """Stop a job step within an allocation.

        :param step_id: id of the step to be stopped
        :type step_id: str
        :raises LauncherError: if unable to stop job step
        """
        pass

    @abc.abstractmethod
    def is_finished(self, status):
        """Based on the statuses gleaned from the workload manager
           determine wether a job is finished or not.

        :param status: status parsed from the wlm
        :type status: str
        :returns: True/False wether job is finished
        :rtype: bool
        """
        pass

    @abc.abstractmethod
    def free_alloc(self, alloc_id):
        """Free an allocation from within the launcher so
           that these resources can be used by other users.

        :param alloc_id: allocation id
        :type alloc_id: str
        :raises LauncherError: if allocation not found within the AllocManager
        :raises LauncherError: if allocation could not be freed
        """
        pass
