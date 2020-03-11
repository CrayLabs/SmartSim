"""
Interface for implementing unique launchers on distributed
systems.
    - Examples: Slurm, pbs pro, Urika-xc, etc

"""

import abc
from subprocess import PIPE, Popen, CalledProcessError
from .launcherUtil import seq_to_str, execute_cmd
import os
import time
import atexit
from ..error import LauncherError

from ..utils import get_logger
logger = get_logger(__name__)


class Launcher(abc.ABC):
    cancel_err_mess = "Unable to revoke your allocation for jobid %s\n"
    run_err_mess = "An error occurred while trying to run the command:\n"

    def __init__(self,
                 def_nodes=1,
                 def_ppn=1,
                 def_partition=None,
                 def_queue=None,
                 def_duration="1:00:00"):
        """ :param def_nodes: Default number of nodes to allocation
            :param def_ppn: Default processes per node
            :param def_partition: Default partition to select
            :param def_queue: Default queue
            :param def_duration: is the default walltime HH:MM:SS
        """
        self.def_nodes = def_nodes
        self.def_ppn = def_ppn
        self.def_partition = def_partition
        self.def_queue = def_queue
        self.alloc_ids = dict()
        self.def_duration = def_duration
        super().__init__()

    #-------------- Abstract Methods --------------
    @abc.abstractmethod
    def validate(self, nodes=None, ppn=None, partition=None):
        """Validate the functionality of the launcher and availability of resources on the system
        :param nodes: Override the number of nodes to validate
        :param ppn: Override the processes per node to validate
        :param partition: Override the partition to validate
        :param verbose: Define verbosity
        :return:
        """
        pass

    @abc.abstractmethod
    def _get_alloc_cmd(self,
                       nodes,
                       ppn,
                       partition,
                       start_time,
                       duration,
                       add_opts,
                       to_string=False,
                       debug=False):
        """
        A method to translate the requested resources into a proper command for making the reservation
        """
        pass

    @abc.abstractmethod
    def get_alloc(self, nodes=None, ppn=None, partition=None, add_opts=None):
        """Get an allocation on the current system using the given launcher interface
        :param nodes: Override the number of nodes to allocate
        :param ppn: Override the number of processes per node to allocate
        :param partition: Override the partition to allocation on
        :param add_opts: Additional options to add tot eh allocation command, e.g salloc, qsub, etc
        :return (int): The allocation id
        """
        pass

    @abc.abstractmethod
    def run_on_alloc(self,
                     cmd,
                     nodes=None,
                     ppn=None,
                     duration="",
                     wait=True,
                     add_opts=None,
                     partition=None,
                     wd=""):
        """
        Runs a command on an allocation.
        The user is expected to have made an allocation before calling this function
        throws badOpsException is the allocation is not made
        For instance the user may have reserved 4 nodes and 10 tasks
        she can submit aprun/mpirun commands and use a parts of her reservation for each command
        """
        pass

    @abc.abstractmethod
    def _get_free_cmd(self, alloc_id):
        """
        returns the workload manager dependant command to release the allocation
        This is used in the free_alloc function which is implemented below.

        :param alloc_id (string): The allocation id to be released.
        """
        pass

    @abc.abstractmethod
    def stop(self, job_id):
        """
        Stops the job with specified job_id.

        :param str job_id: The job indentifier
        """
        pass

    def free_alloc(self, alloc_id):
        """
        if there is an active researvation, or if alloc_id is specified it gets cancelled
        """
        if alloc_id not in self.alloc_ids.keys():
            raise LauncherError("Allocation id, " + str(alloc_id) +
                                " not found.")

        (cancel_cmd, cancel_err_mess) = self._get_free_cmd(alloc_id)
        returncode, _, err = execute_cmd(cancel_cmd)

        if returncode != 0:
            logger.info("Unable to revoke your allocation for jobid %s" % alloc_id)
            logger.info(
                "The job may have already timed out, or you may need to cancel the job manually")
            raise LauncherError("Unable to revoke your allocation for jobid %s" % alloc_id)

        logger.info("Successfully freed allocation %s" % alloc_id)
        self.alloc_ids.pop(alloc_id)
