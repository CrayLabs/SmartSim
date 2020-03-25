import calendar
import time
import os
import atexit
import sys

from .launcher import Launcher
from subprocess import PIPE, Popen, CalledProcessError
from .launcherUtil import seq_to_str, execute_cmd, write_to_bash
from ..error import LauncherError

from ..utils import get_logger, get_env
logger = get_logger(__name__)


class SlurmLauncher(Launcher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    class ComputeNode():

        def __init__(self, node_name=None, node_ppn=None):
            self.name = node_name
            self.ppn = node_ppn

        def _is_valid_node(self):
            if self.name is None:
                return False
            if self.ppn is None:
                return False

            return True

    class Partition():

        def __init__(self):
            self.name = None
            self.min_ppn = None
            self.nodes = set()

        def _is_valid_partition(self):

            if self.name is None:
                return False
            if len(self.nodes)<=0:
                return False
            for node in self.nodes:
                if not node._is_valid_node():
                    return False

            return True

    def sstat(self, args):
        """Calls sstat with args
           :params args: List of command and command arguments
           :type args: List of str
           :returns: Output and error of sstat
        """

        cmd = ["sstat"] + args
        returncode, out, error = execute_cmd(cmd)
        return out, error

    def sacct(self, args):
        """Calls sacct with args

           :params args: List of command and command arguments
           :type args: List of str
           :returns: Output and error of sacct
        """
        cmd = ["sacct"] + args
        returncode, out, error = execute_cmd(cmd)
        return out, error

    def salloc(self, args):
        """Calls slurm salloc with args
           :params args: List of command and command arguments
           :type args: List of str
           :returns: Output and error of salloc
        """

        cmd = ["salloc"] + args
        returncode, out, error = execute_cmd(cmd)
        return out, error

    def sinfo(self, args):
        """Calls slurm sinfo with args
           :params args: List of command and command arguments
           :type args: List of str
           :returns: Output and error of sinfo
        """

        cmd = ["sinfo"] + args
        returncode, out, error = execute_cmd(cmd)
        return out, error


    def _get_system_partition_info(self):
        """Build a dictionary of slurm partitions filled with
           Partition objects.
           :returns: dict of Partition objects
        """

        sinfo_output, sinfo_error = self.sinfo(["--noheader", "--format", "%R %n %c"])

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
                partitions.update({p_name:self.Partition()})

            partitions[p_name].name = p_name
            partitions[p_name].nodes.add(self.ComputeNode(node_name=p_node, node_ppn=p_ppn))

        return partitions

    def _get_default_partition(self):
        """Returns the default partition from slurm which is the partition with
           a star following its partition name in sinfo output
        """
        sinfo_output, sinfo_error = self.sinfo(["--noheader", "--format", "%P"])

        default = None
        for line in sinfo_output.split("\n"):
            if line.endswith("*"):
                default = line.strip("*")

        if not default:
            raise LauncherError("Could not find default partition!")
        return default

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
            nodes = self.def_nodes
        if not ppn:
            ppn = self.def_ppn

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


    def get_job_stat(self, job_id):
        """
        given a job_id returns the status and return code
        :param job_id (string or castable to string) the id of the job to be queried
        :return a string tuple (job_status, return_code)
        """
        job_id = str(job_id)
        sacct_out, sacct_error = self.sacct(["--noheader", "-p", "-b", "-j", job_id])
        if not sacct_out:
            result = ("NOTFOUND","NAN")
        else:
            for line in sacct_out.split("\n"):
                if line.strip().startswith(job_id):
                    line = line.split("|")
                    stat = line[1]
                    code = line[2].split(':')[0]
                    result = (stat, code)
                    break
            else:
                result = ("NOTFOUND","NAN")
        return result


    def get_job_nodes(self, job_id):
        """ given a job_id, returns the list of nodes that job is running on
            :param: job_id (string or castable to string) the id of the job to be queried
            :return a list of strings corresponding to nodes
        """
        jid = str(job_id)
        output, error = self.sstat([jid, "-i", "-n", "-p", "-a"])
        nodes = []
        if "error:" in error.split(" "):
            raise LauncherError("Could not find allocation for job: " + job_id)
        else:
            for line in output.split("\n"):
                sstat_string = line.split("|")

                # sometimes there are \n that we need to ignore
                if len(sstat_string) >= 2:
                    node = sstat_string[1]
                    nodes.append(node)

        # remove duplicates
        nodes = list(set(nodes))
        return nodes

    def accept_allocation(self, alloc_id, partition, nodes):
        """Take in an allocation provided by the user

           :param str alloc_id: ID of the allocation
           :param str partition: partition of the allocation aquired by the user
           :param int nodes: number of nodes in the allocation
        """
        if partition in self.alloc_manager:
            err_msg = "SmartSim does not allow two allocations on the same partition"
            err_msg += f"\n Allocation {self.alloc_manager[partition]} already allocated"
            raise LauncherError(err_msg)
        else:
            self.alloc_manager.add_alloc(alloc_id, partition, nodes)


    def get_alloc(self, nodes=None, ppn=None, partition=None, start_time=None, duration="",
        add_opts=None):
        """ Get the allocation using salloc and specified constraints.

            :param nodes: Override the default node count to allocate
            :param ppn: Override the default processes per node to allocate
            :param partition: Override the default partition to allocate from
            :param add_opts: Additional options to pass to salloc
            :return: allocation id
        """
        if partition == "default":
            partition = None
        if partition in self.alloc_manager:
            err_msg = "SmartSim does not allow two allocations on the same partition"
            err_msg += f"\n Allocation {self.alloc_manager[partition]} already allocated"
            raise LauncherError(err_msg)
        if not nodes:
            nodes = self.def_nodes
        if not ppn:
            ppn = self.def_ppn


        salloc = self._get_alloc_cmd(nodes, ppn, partition, start_time,
                                    duration, add_opts, to_string=False)
        logger.debug(salloc)
        logger.debug("allocting %d nodes on partition %s" %(nodes, partition))

        #TODO figure out why this goes to stderr
        returncode, _, err = execute_cmd(salloc)
        alloc_id = self._parse_salloc(err)
        if alloc_id:
            logger.info("Allocation successful with Job ID: %s" % alloc_id)
            self.alloc_manager.add_alloc(alloc_id, partition, nodes)
        else:
            error = self._parse_salloc_error(err)
            raise LauncherError(error)
        return str(alloc_id)


    def run_on_alloc(self, cmd, alloc_id, nodes=None, ppn=None, duration="", add_opts=None,
                    partition=None, cwd="", env_vars=None, out_file=None, err_file=None):
        """Build and call "srun" on a user provided command within an allocation.

           :param cmd: command to run with "srun"
           :type cmd: list of strings
           :param int alloc_id: allocation id to run command on
           :param int nodes: number of nodes
           :param int ppn: number of processes per node
           :param str duration: time of job in hour:min:second format e.g. 10:00:00
           :param add_opts: additional options for the "srun" command
           :type add_opts: list of strings
           :param str partition: partition to run job on
           :param str cwd: current working directory to launch srun in
           :param env_vars: environment variables to pass to the srun command
           :type env_vars: dict of environment variables
           :param str out_file: file to capture output of srun command
           :param str err_file: file to capture error of srun command
           :return: subjob id
           :raises: LauncherError
        """
        if str(alloc_id) not in self.alloc_manager().keys():
            raise LauncherError("Could not find allocation with id: " + str(alloc_id))
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        if not add_opts:
            add_opts = []
        if not nodes:
            nodes = self.def_nodes
        if not ppn:
            ppn = self.def_ppn
        if not partition:
            partition = self.def_partition
        ntasks = ppn * nodes
        if not cwd:
            cwd = os.getcwd()
        if not out_file:
            out_file = repr(self.alloc_manager()[alloc_id]) + ".out"
            out_file = os.path.join(cwd, out_file)
        if not err_file:
            err_file = repr(self.alloc_manager()[alloc_id]) + ".err"
            err_file = os.path.join(cwd, err_file)

        srun = ["srun", "--jobid", str(alloc_id),
                        "--nodes", str(nodes),
                        "--ntasks", str(ntasks),
                        "--output", out_file,
                        "--error", err_file]

        if duration:
            srun += ["-t", duration]
        if partition:
            srun += ["--partition", partition]
        if env_vars:
            env_var_str = self._format_env_vars(env_vars)
            srun += ["--export", env_var_str]

        if len(add_opts) > 0 and isinstance(add_opts, list):
            for opt in add_opts:
                srun.append(opt)
        srun += [cmd]
        logger.debug(seq_to_str(srun))

        status = self._run_asynch_command(seq_to_str(srun), cwd)
        if status == -1:
            raise LauncherError("Failed to run on allocation")
        else:
            return self.alloc_manager.add_subjob(alloc_id)

    def stop(self, job_id):
        """Stop is used to stop a subjob that is currently running within an allocation.

            :param str job_id: sub job id with decimal increment to stop e.g. 64253.1
        """
        status, _ = self.get_job_stat(job_id)
        if status != "COMPLETE":
            cancel_cmd = ["scancel", job_id]
            returncode, output, err = execute_cmd(cancel_cmd)

            if returncode != 0:
                raise LauncherError("Unable to stop jobid %s" % job_id)
            else:
                logger.info("Successfully stopped job %s" % job_id)

    def _get_free_cmd(self, alloc_id):
        scancel = ["scancel", alloc_id]
        err_mess = "Unable to revoke your allocation for jobid %s\n"\
                    "The job may have already timed out, or you may need to cancel the job manually" % alloc_id
        return scancel, err_mess

    def _parse_salloc(self, output):
        for line in output.split("\n"):
            if line.startswith("salloc: Granted job allocation"):
                return line.split()[-1]

    def _parse_salloc_error(self, output):
        for line in output.split("\n"):
            if line.startswith("salloc:"):
                error = line.split("error:")[1]
                return error

    def _get_alloc_cmd(self, nodes, ppn, partition, start_time, duration, add_opts, to_string=False):
        logger.debug("requesting %d nodes and %d tasks" %(nodes, ppn* nodes))
        salloc = ["salloc", "--no-shell", "-N", str(nodes), "--ntasks", str(nodes * ppn)]
        if add_opts and isinstance(add_opts, list):
            salloc += add_opts
        if partition is not None:
            salloc += ["--partition", partition]
        if start_time:
            salloc += ["--begin", start_time]
        if duration:
            salloc += ["--time", duration]
        salloc += ["-J SmartSim"]
        if to_string:
            return seq_to_str(salloc, add_equal=True)
        return salloc

    def _run_asynch_command(self, cmd, cwd):
        try:
            popen_obj = Popen(cmd, cwd=cwd, shell=True)
        except OSError as err:
            logger.error(err)
            return -1
        return 1

    def _format_env_vars(self, env_vars):
        """Slurm takes exports in comma seperated lists
           the list starts with all as to not disturb the rest of the environment
           for more information on this, see the slurm documentation for srun"""
        path = get_env("PATH")
        python_path = get_env("PYTHONPATH")
        format_str = "".join(("PATH=", path, ",", "PYTHONPATH=", python_path))

        for k, v in env_vars.items():
            format_str += "," + "=".join((k,v))
        return format_str

    def is_finished(self, status):
        """Determines wether or not a job is finished based on the Slurm status"""

        terminals = [
            "PENDING",
            "COMPLETING",
            "PREEMPTED",
            "RUNNING",
            "SUSPENDED",
            "STOPPED",
            "NOTFOUND"
            ]

        if status in terminals:
            return False
        return True
