import calendar
import time
import os
import atexit
import sys

from .launcher import Launcher
from subprocess import PIPE, Popen, CalledProcessError
from .launcherUtil import seq_to_str, execute_cmd, write_to_bash
from ..error import LauncherError

from ..utils import get_logger
logger = get_logger(__name__)


class SlurmLauncher(Launcher):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.subjob_ids = list()
		self.alloc_partition = None

	class Partition():

		def __init__(self):
			self.name = None
			self.min_ppn = None
			self.nodes = set()

		def _is_valid_partition(self):

			if self.name is None:
				return False
			if self.min_ppn is None:
				return False
			if self.min_ppn<=0:
				return False
			if len(self.nodes)<=0:
				return False

			return True

	def sstat(self, args):
		"""Calls sstat sstat with args
		   :params args: List of command and command arguments
		   :type args: List of str
		   :returns: Output str of sstat command with args
		"""

		cmd = ["sstat"] + args
		cmd_err_mess = "Could not execute sstat!"

		cmd_output_str, _ = execute_cmd(cmd, err_message=cmd_err_mess)

		return cmd_output_str

	def salloc(self, args):
		"""Calls slurm salloc with args
		   :params args: List of command and command arguments
		   :type args: List of str
		   :returns: Output str of salloc command with args
		"""

		cmd = ["salloc"] + args
		cmd_err_mess = "Could not execute salloc!"

		cmd_output_str, _ = execute_cmd(cmd, err_message=cmd_err_mess)

		return cmd_output_str

	def sinfo(self, args):
		"""Calls slurm sinfo with args
		   :params args: List of command and command arguments
		   :type args: List of str
		   :returns: Output str of sinfo command with args
		"""

		cmd = ["sinfo"] + args
		cmd_err_mess = "Could not execute sinfo!"

		cmd_output_str, _ = execute_cmd(cmd, err_message=cmd_err_mess)

		return cmd_output_str

	def _get_system_partition_info(self):
		"""Build a dictionary of slurm partitions filled with
		   Partition objects.
		   :returns: dict of Partition objects
		"""

		sinfo_output = self.sinfo(["--noheader", "--format", "%R %n %c"])

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
			partitions[p_name].nodes.add(p_node)
			if not partitions[p_name].min_ppn:
				partitions[p_name].min_ppn = p_ppn
			partitions[p_name].min_ppn = min(p_ppn, partitions[p_name].min_ppn)

		return partitions

	def _get_default_partition(self):
		"""Returns the default partition from slurm which is the partition with
           a star following its partition name in sinfo output
        """
		sinfo_output = self.sinfo(["--noheader", "--format", "%P"])

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
			:raises: LauncerError
		"""

		sys_partitions = self._get_system_partition_info()

		n_avail_nodes = 0
		avail_nodes = set()
		min_ppn = None

		p = partition
		if not partition:
			p = self._get_default_partition()

		if not p in sys_partitions:
			raise LauncherError("Partition {0} is not found on this system".format(p))
		avail_nodes = avail_nodes.union(sys_partitions[p].nodes)
		if not min_ppn:
			min_ppn = sys_partitions[p].min_ppn
		else:
			min_ppn = min(sys_partitions[p].min_ppn, min_ppn)

		n_avail_nodes = len(avail_nodes)
		logger.debug("Found {0} nodes that match the constraints provided".format(n_avail_nodes))
		if n_avail_nodes<nodes:
			raise LauncherError("{0} nodes are not available on the specified partitions.  Only "\
								"{1} nodes available.".format(nodes,n_avail_nodes))
		if min_ppn < ppn:
			raise LauncherError("{0} ppn is not available on each node.".format(min_ppn))

		logger.info("Successfully validated Slurm with sufficient resources")

	def get_job_stat(self, job_id):
		"""
		given a job_id returns the status and return code
		:param job_id (string or castable to string) the id of the job to be queried
		:return a string tuple (job_status, return_code)
		"""
		job_id = str(job_id)
		output,_ = execute_cmd(["sjobexitmod", "-l", job_id])
		if not output:
			result = ("NOTFOUND","NAN")
		else:
			for line in output.split("\n"):
				if line.strip().startswith(job_id):
					line = line.split()
					stat =line[3]
					code = line[4].split(':')[0]
					result = (stat, code)
					break
			else:
				result = ("NOTFOUND","NAN")
		return result

	def get_sjob_stat(self, sub_job_id):
		"""Use the get_job_nodes function to determine the status of a job. If an error is not
		   raised when attempting to retrive the nodes, then the job is still running.
		   :param str sub_job_id: job id of the sub job of self.alloc
		   :returns: 1 if job is running -1 otherwise
		"""
		try:
			nodes = self.get_job_nodes(sub_job_id)
			return 1
		except LauncherError:
			return -1

	def get_job_nodes(self, job_id):
		""" given a job_id, returns the list of nodes that job is running on
			:param: job_id (string or castable to string) the id of the job to be queried
			:return a list of strings corresponding to nodes
		"""
		jid = str(job_id)
		output, error = execute_cmd(["sstat", jid, "-i", "-n", "-p", "-a"])
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

	def get_alloc(self, nodes=None, ppn=None, partition=None, start_time=None, duration="",
		add_opts=None):
		""" Get the allocation using salloc and specified constraints.

			:param nodes: Override the default node count to allocate
			:param ppn: Override the default processes per node to allocate
			:param partition: Override the default partition to allocate from
			:param add_opts: Additional options to pass to salloc
			:return: allocation id
		"""
		if not nodes:
			nodes = self.def_nodes
		if not ppn:
			ppn = self.def_ppn

		salloc = self._get_alloc_cmd(nodes, ppn, partition, start_time,
									duration, add_opts, to_string=False)
		logger.debug("allocting %d nodes %d tasks/node, partition %s" %(nodes, ppn, partition))

		_, err = execute_cmd(salloc)
		alloc_id = self._parse_salloc(err)
		if alloc_id:
			logger.info("Allocation Successful with Job ID: %s" % alloc_id)
			# start sub_jobid counter
			self.alloc_ids[alloc_id] = 0
		else:
			raise LauncherError("Failed to get requested allocation")
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
		if str(alloc_id) not in self.alloc_ids.keys():
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
			out_file = "-".join((str(alloc_id), str(self.alloc_ids[alloc_id]) + ".out"))
			out_file = os.path.join(cwd, out_file)
		if not err_file:
			err_file = "-".join((str(alloc_id), str(self.alloc_ids[alloc_id]) + ".err"))
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
			subjob_id = ".".join((alloc_id, str(self.alloc_ids[alloc_id])))
			self.alloc_ids[alloc_id] += 1
			self.subjob_ids.append(subjob_id)
			return subjob_id

	def stop(self, job_id):
		"""Stop is used to stop a subjob that is currently running within an allocation.

			:param str job_id: sub job id with decimal increment to stop e.g. 64253.1
        """
		if job_id not in self.subjob_ids:
			raise LauncherError("Job id, " + str(job_id) + " not found.")

		(cancel_cmd, cancel_err_mess) = self._get_free_cmd(job_id)
		try:
			_, err = execute_cmd(cancel_cmd, err_message=cancel_err_mess)

		except CalledProcessError:
			logger.info("Unable to cancel jobid %s" % job_id)
			raise LauncherError(cancel_err_mess)

		logger.info("Successfully canceled job %s" % job_id)
		self.subjob_ids.remove(job_id)

	def _get_free_cmd(self, alloc_id):
		scancel = ["scancel", alloc_id]
		err_mess = "Unable to revoke your allocation for jobid %s\n"\
					"The job may have already timed out, or you may need to cancel the job manually" % alloc_id
		return scancel, err_mess

	def _parse_salloc(self, output):
		for line in output.split("\n"):
			if line.startswith("salloc: Granted job allocation"):
				return line.split()[-1]

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
			logger.debug(err)
			return -1
		return 1

	def _format_env_vars(self, env_vars):
		"""Slurm takes exports in comma seperated lists
		   the list starts with all as to not disturb the rest of the environment
           for more information on this, see the slurm documentation for srun"""
		format_str = "ALL"
		for k, v in env_vars.items():
			format_str += "," + "=".join((k,v))
		return format_str