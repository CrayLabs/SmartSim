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
		self.alloc_partition = None

	def validate(self, nodes=None, ppn=None, partition=None):
		"""Validate slurm on the system and check for sufficient resources using 'sinfo'
			:param nodes: Override the default node count to validate
			:param ppn: Override the default processes per node to validate
			:param partition: Override the default partition to validate
			:return:
		"""
		if not nodes:
			nodes = self.def_nodes
		if not ppn:
			ppn = self.def_ppn
		if not partition:
			partition = self.def_partition

		sinfo_cmd = ["sinfo", "--noheader", "--format", "%R %D %c"]
		err_mess = "Failed to validate slurm!"

		out_str, _ = execute_cmd(sinfo_cmd, err_message=err_mess)

		total_nodes = 0
		for line in out_str.split("\n"):
			line  = line.strip()
			if line == "":
				continue

			partition_details = line.split(" ")
			if partition is None or partition_details[0] == partition:
				# Strip + character from core count if detected
				if not partition_details[2][-1].isnumeric():
					partition_details[2] = partition_details[2][:-1]
				if int(partition_details[2]) >= ppn:
					total_nodes += int(partition_details[1])

		logger.debug("Found %d nodes that match the constraints provided" % total_nodes)

		if total_nodes < nodes:
			raise LauncherError("Could not find enough nodes with the specified constraints: \n"
												"Nodes Requested: %i\n"
												"Nodes Detected: %i" % (nodes, total_nodes))

		logger.info("Successfully Validated Slurm with sufficient resources")

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
		if "error:" in output.split(" "):
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
			The reason we override the fields below is that it becomes the default for submitting jobs
			against this allocation!
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

		if len(add_opts) > 0:
			for opt in add_opts:
				srun.append(opt)
		srun += [cmd]

		status = self._run_asynch_command(seq_to_str(srun), cwd)
		if status == -1:
			raise LauncherError("Failed to run on allocation")
		else:
			subjob_id = ".".join((alloc_id, str(self.alloc_ids[alloc_id])))
			self.alloc_ids[alloc_id] += 1
			return subjob_id

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