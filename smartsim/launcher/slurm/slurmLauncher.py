import calendar
import time
import os
import atexit
import sys
import numpy as np

from shutil import which
from ..launcher import Launcher
from ..launcherUtil import seq_to_str, write_to_bash, ComputeNode, Partition
from ...error import LauncherError, SSConfigError
from .slurmAllocation import SlurmAllocation
from .slurmParser import parse_sacct, parse_sacct_step, parse_salloc
from .slurmParser import parse_salloc_error, parse_sstat_nodes, parse_step_id_from_sacct
from .slurmStep import SlurmStep
from .slurm import sstat, sacct, salloc, sinfo, scancel
from ..shell import execute_cmd, execute_async_cmd, is_remote

from ...utils import get_logger, get_env
logger = get_logger(__name__)


class SlurmLauncher(Launcher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self, nodes=None, ppn=None, partition=None):
        """Check that there are sufficient resources in the provided Slurm partitions.
           :param str partition: partition to validate
           :param nodes: Override the default node count to validate
           :type nodes: int
           :param ppn: Override the default processes per node to validate
           :type ppn: int
           :raises: LauncherError
        """
        sys_partitions = self._get_system_partition_info()

        n_avail_nodes = 0
        avail_nodes = set()

        if not nodes:
            nodes = 1
        if not ppn:
            ppn = 1

        p_name = partition
        if p_name is None or p_name == "default":
            p_name = self._get_default_partition()

        if not p_name in sys_partitions:
            raise LauncherError("Partition {0} is not found on this system".format(p_name))

        for node in sys_partitions[p_name].nodes:
            if node.ppn >= ppn:
                avail_nodes.add(node)

        n_avail_nodes = len(avail_nodes)
        logger.debug("Found {0} nodes that match the constraints provided".format(n_avail_nodes))
        if n_avail_nodes<nodes:
            raise LauncherError("{0} nodes are not available on the specified partitions.  Only "\
                                "{1} nodes available.".format(nodes,n_avail_nodes))

        logger.info("Successfully validated Slurm with sufficient resources")

    def create_step(self, entity_name, run_settings, multi_prog=False):
        """Convert a smartsim entity run_settings into a Slurm step
           to be launched on an allocation. An entity must have an
           allocation assigned to it in the running settings or
           create_step will throw a LauncherError

        :param entity_name: name of the entity to create a step for
        :type entity_name: str
        :param run_settings: smartsim run settings for an entity
        :type run_settings: dict
        :param multi_prog: Create step with slurm --multi-prog, defaults to False
        :type multi_prog: bool, optional
        :raises LauncherError: if no allocation specified for the step
        :return: slurm job step
        :rtype: SlurmStep
        """
        try:
            if "alloc" not in run_settings:
                raise SSConfigError(f"User provided no allocation for {entity_name}")

            alloc_id = run_settings["alloc"]
            if alloc_id not in self.alloc_manager():
                raise LauncherError(
                    f"Allocation {alloc_id} has not been obtained or added to SmartSim")
            else:
                # increment the internal job per allocation counter
                step_num = self.alloc_manager.add_step(str(alloc_id))
                name = entity_name + '-' + str(np.base_repr(time.time_ns(), 36))
                step = SlurmStep(name, run_settings, multi_prog)
                return step
        except SSConfigError as e:
            raise LauncherError("Job step creation failed: " + e.msg) from None

    def get_step_status(self, step_id):
        """Get the status of a SlurmStep via the id of the step
           e.g. 12345.0

        :param step_id: id of the step in form xxxxx.x
        :type step_id: str
        :return: status of the job and returncode
        :rtype: tuple of (str, str)
        """
        step_id = str(step_id)
        sacct_out, sacct_error = sacct(["--noheader", "-p", "-b", "-j", step_id])
        if not sacct_out:
            result = ("NOTFOUND","NAN")
            return result
        else:
            return parse_sacct(sacct_out, step_id)

    def get_step_nodes(self, step_id):
        """Return the compute nodes of a specific job or allocation
           in a list with the duplicates removed.

        :param step_id: job step id or allocation id
        :type step_id: str
        :raises LauncherError: if allocation or job step cannot be
                               found
        :return: list of compute nodes the job was launched on
        :rtype: list of str
        """
        step_id = str(step_id)
        output, error = sstat([step_id, "-i", "-n", "-p", "-a"])
        if "error:" in error.split(" "):
            raise LauncherError("Could not find allocation for job: " + step_id)
        else:
            return parse_sstat_nodes(output)

    def accept_alloc(self, alloc_id):
        """Accept a user provided and obtained allocation into the
           Launcher for future launching of entities. Obtain as much
           information about the allocation as possible by parsing the
           output of slurm commands.

        :param alloc_id: id of the allocation
        :type alloc_id: str
        :raises LauncherError: if the allocation cannot be found
        """
        self._check_for_slurm()

        alloc_id = str(alloc_id)
        sacct_out, sacct_error = sacct(["--noheader", "-p",
                                        "-b", "-j", alloc_id])
        if not sacct_out:
            raise LauncherError(
                f"User provided allocation, {alloc_id}, could not be found")

        nodes = len(self.get_step_nodes(alloc_id))
        allocation = SlurmAllocation(nodes=nodes)
        self.alloc_manager.add(alloc_id, allocation)

    def get_alloc(self, nodes=1, ppn=1, duration="1:00:00", **kwargs):
        """Request an allocation with the specified arguments. Anything
           passed to the keywords args will be processed as a Slurm
           argument and appended to the salloc command with the appropriate
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
        self._check_for_slurm()

        allocation = SlurmAllocation(nodes=nodes, ppn=ppn,
                                     duration=duration, **kwargs)
        salloc = allocation.get_alloc_cmd()
        logger.debug(salloc)

        #TODO figure out why this goes to stderr
        returncode, _, err = execute_cmd(salloc)
        alloc_id = parse_salloc(err)
        if alloc_id:
            logger.info("Allocation successful with Job ID: %s" % alloc_id)
            self.alloc_manager.add(alloc_id, allocation)
        else:
            error = parse_salloc_error(err)
            raise LauncherError(error)
        return str(alloc_id)

    def run(self, step):
        """Run a job step on an allocation through the slurm launcher.
           A constructed job step is required such that the argument
           translation from SmartSimEntity to SlurmStep has been
           completed and an allocation has been assigned to the step.

        :param step: Job step to be launched
        :type step: SlurmStep
        :raises LauncherError: If the allocation cannot be found or the
                               job step failed to launch.
        :return: job_step id
        :rtype: str
        """
        self._check_for_slurm()

        alloc_id = step.alloc_id
        if str(alloc_id) not in self.alloc_manager().keys():
            raise LauncherError("Could not find allocation with id: " + str(alloc_id))
        srun = step.build_cmd()

        status, _, _ = execute_async_cmd([seq_to_str(srun)], cwd=step.cwd)
        if status == -1:
            raise LauncherError("Failed to run on allocation")
        else:
            return self._get_slurm_step_id(step)

    def stop(self, step_id):
        """Stop a job step within an allocation.

        :param step_id: id of the step to be stopped
        :type step_id: str
        :raises LauncherError: if unable to stop job step
        """
        self._check_for_slurm()

        status, _ = self.get_step_status(step_id)
        if status != "COMPLETE":
            returncode, out, err = scancel([str(step_id)])
            if returncode != 0:
                raise LauncherError(f"Unable to stop job {str(step_id)}")
            else:
                logger.info(f"Successfully stopped job {str(step_id)}")

    def is_finished(self, status):
        """Determine wether a job is finished according to slurm
           status parsed from sacct

        :param status: status returned from sacct command
        :type status: str
        :return: True/False wether job is finished
        :rtype: bool
        """

        terminals = [
            "PENDING", "COMPLETING", "PREEMPTED",
            "RUNNING", "SUSPENDED", "STOPPED",
            "NOTFOUND"]

        if status in terminals:
            return False
        return True

    def free_alloc(self, alloc_id):
        """Free an allocation from within the launcher so
           that these resources can be used by other users.

        :param alloc_id: allocation id
        :type alloc_id: str
        :raises LauncherError: if allocation not found within the AllocManager
        :raises LauncherError: if allocation could not be freed
        """
        self._check_for_slurm()

        if alloc_id not in self.alloc_manager().keys():
            raise LauncherError("Allocation id, " + str(alloc_id) +
                                " not found.")

        returncode, _, err = scancel([str(alloc_id)])

        if returncode != 0:
            logger.info("Unable to revoke your allocation for jobid %s" % alloc_id)
            logger.info(
                "The job may have already timed out, or you may need to cancel the job manually")
            raise LauncherError("Unable to revoke your allocation for jobid %s" % alloc_id)

        self.alloc_manager.remove(alloc_id)
        logger.info("Successfully freed allocation %s" % alloc_id)


    def _get_slurm_step_id(self, step):
        """Use the sacct command to find the step_id of a step that
           has been launched by this SlurmLauncher instance. Use the
           step name to find the corresponding step_id.

           Perform a number of trials in case the slurm launcher
           takes a while to populate the Sacct database

        :param step: step to find the id of
        :type step: SlurmStep
        :returns: if of the step
        :rtype: str
        """
        time.sleep(1)
        n_trials = 10
        step_id = None
        while n_trials > 0:
            output, error = sacct(["--noheader", "-p",
                                   "--format=jobname,jobid",
                                   "--job=" + step.alloc_id])
            step_id = parse_step_id_from_sacct(output, step.name)
            if step_id:
                break
            else:
                time.sleep(1)
                n_trials -= 1
        if not step_id:
            raise LauncherError("Could not find id of launched job step")
        return step_id


    def _get_system_partition_info(self):
        """Build a dictionary of slurm partitions filled with
           Partition objects.
           :returns: dict of Partition objects
        """

        sinfo_output, sinfo_error = sinfo(["--noheader", "--format", "%R %n %c"])

        partitions = {}
        for line in sinfo_output.split("\n"):
            line  = line.strip()
            if line == "":
                continue

            p_info = line.split(" ")
            p_name = p_info[0]
            p_node = p_info[1]
            p_ppn = int(p_info[2])

            if not p_name in partitions:
                partitions.update({p_name:Partition()})

            partitions[p_name].name = p_name
            partitions[p_name].nodes.add(ComputeNode(node_name=p_node, node_ppn=p_ppn))

        return partitions

    def _get_default_partition(self):
        """Returns the default partition from slurm which is the partition with
           a star following its partition name in sinfo output
        """
        sinfo_output, sinfo_error = sinfo(["--noheader", "--format", "%P"])

        default = None
        for line in sinfo_output.split("\n"):
            if line.endswith("*"):
                default = line.strip("*")

        if not default:
            raise LauncherError("Could not find default partition!")
        return default


    def _check_for_slurm(self):
        """Check for slurm if not using a remote Command Server and return
           an error if the user has not initalized the remote launcher.

        :raises LauncherError: if no access to slurm and no remote Command
                               Server has been initialized.
        """
        if not which("salloc") and not is_remote():
            error = "User attempted Slurm methods without access to Slurm at the call site.\n"
            error += "Setup a Command Server, and initialize with"
            error += " SmartSim.remote.init_command_server()"
            raise LauncherError(error)
