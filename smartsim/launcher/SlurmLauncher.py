import calendar
import time
import os
import atexit

from .launcher import Launcher
from subprocess import PIPE, Popen, CalledProcessError
from .launcherUtil import LauncherArgException, seq_to_str, LauncherBadOpException
from .launcherUtil import get_output_err, prepend_cd

from ..utils import get_logger
logger = get_logger(__name__)


class SlurmLauncher(Launcher):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.alloc_partition = None
		self._script_submit_command = "sbatch"
		self._pending_script = False

	def validate(self, nodes=None, ppn=None, partition=None, debug=False):
		"""Validate slurm on the system and check for sufficient resources using 'sinfo'
			:param nodes: Override the default node count to validate
			:param ppn: Override the default processes per node to validate
			:param partition: Override the default partition to validate
			:param verbose: Set verbosity
			:return:
		"""
		if nodes is None:
			nodes = self.def_nodes
		if ppn is None:
			ppn = self.def_ppn
		if partition is None:
			partition = self.def_partition

		sinfo_cmd = ["sinfo", "--noheader", "--format", "%R %D %c"]
		err_mess = "Failed to validate slurm!"

		out_str, _ = Launcher.execute_cmd(sinfo_cmd, err_message=err_mess)
		if debug:
			print(out_str)
		# Parse the output string to extract the available resources for the specified partition
		# Example output from sinfo --noheader --format %R %D %c:
		# bdw18 32 72
		# bdw10 9 40
		# npl24 2 96
		# ...
		tot_nodes = 0
		for line in out_str.split("\n"):

			# there are empty lines that we need to skip!
			line  = line.strip()
			if line == "":
				continue

			partition_details = line.split(" ")
			if partition is None or partition_details[0] == partition:
				# Strip + character from core count if detected
				if not partition_details[2][-1].isnumeric():
					partition_details[2] = partition_details[2][:-1]
				if int(partition_details[2]) >= ppn:
					tot_nodes += int(partition_details[1])

		logger.debug("Found %d nodes that match the constraints provided" % tot_nodes)

		if tot_nodes < nodes:
			raise SystemError("Could not find enough nodes with the specified constraints: \n"
												"Nodes Requested: %i\n"
												"Nodes Detected: %i" % (nodes, tot_nodes))

		logger.info("Successfully Validated Slurm with sufficient resources")

	def _parse_salloc(self, output):
		# remember salloc's output is sent to stderr!
		for line in output.split("\n"):
			if line.startswith("salloc: Granted job allocation"):
				return line.split()[-1]

	@staticmethod
	def get_job_stat(job_id):
		"""
		given a job_id returns the status and return code
		:param job_id (string or castable to string) the id of the job to be queried
		:return a string tuple (job_status, return_code)
		"""
		job_id = str(job_id)
		output,_ = Launcher.execute_cmd(["sjobexitmod", "-l", job_id])
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


	@staticmethod
	def get_job_nodes(job_id):
		""" given a job_id, returns the list of nodes that job is running on
			:param: job_id (string or castable to string) the id of the job to be queried
			:return a list of strings corresponding to nodes
		"""
		jid = str(job_id)
		output, error = Launcher.execute_cmd(["sstat", jid, "-i", "-n", "-p", "-a"])
		nodes = []
		if "error:" in output.split(" "):
			raise SystemError("Could not find allocation for job: " + job_id)
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

	def _get_alloc_cmd(self, nodes, ppn, partition, start_time, duration, add_opts, to_string=False, debug=False):
		print("requesting %d nodes and %d tasks" %(nodes, ppn* nodes))
		salloc = ["salloc", "--no-shell", "-N", str(nodes), "--ntasks", str(nodes * ppn)]
		if add_opts and isinstance(add_opts, list):
			salloc += add_opts
		if partition is not None:
			salloc += ["--partition", partition]
		# If the allocation is made for a later time!
		if start_time:
			salloc += ["--begin", start_time]
		#if not specified the default partition duration will be used
		if duration:
			salloc += ["--time", duration]
		if debug:
			logger.info("allocting %d nodes %d tasks/node, partition %s" %(nodes, ppn, partition))
		if to_string:
			return seq_to_str(salloc, add_equal=True)
		return salloc

	def get_alloc(self, nodes=None, ppn=None, partition=None, start_time=None, duration="",
		add_opts=None, debug=False):
		""" Get the allocation using salloc and specified constraints.
			The reason we override the fields below is that it becomes the default for submitting jobs
			against this allocation!
			:param nodes: Override the default node count to allocate
			:param ppn: Override the default processes per node to allocate
			:param partition: Override the default partition to allocate from
			:param add_opts: Additional options to pass to salloc
			:return:
		"""
		if nodes:
			self.def_nodes = nodes
		else:
			nodes = self.def_nodes

		if ppn:
			self.def_ppn = ppn
		else:
			ppn = self.def_ppn

		salloc = self._get_alloc_cmd(nodes, ppn, partition, start_time,
									duration, add_opts, to_string=False, debug=debug)
		#Salloc's output is redirected to stderr
		_, err = Launcher.execute_cmd(salloc)
		self.alloc_id = self._parse_salloc(err)
		if self.alloc_id is not None:
			logger.info("Allocation Successful with Job ID: %s" % self.alloc_id)
		else:
			return -1
		self.alloc_ids.add(self.alloc_id)
		return int(self.alloc_id)

	def _get_free_cmd(self, alloc_id):
		scancel = ["scancel", alloc_id]
		err_mess = "Unable to revoke your allocation for jobid %s\n"\
					"The job may have already timed out, or you may need to cancel the job manually" % alloc_id
		return scancel, err_mess


	def run(self, cmd=None, use_script=True, wait=True, add_opts=None, wd="", track_output=True):
		"""
		This function defines the default behavior of the launcher with minimum adjustability
		The default behavior is to generate a script and submit it.
		For more control use either make_script & run_script or get_alloc & run_on_alloc
		"""
		if use_script:
			return self.run_script(cmd=cmd, wait=wait, add_opts=add_opts, wd=wd, track_output=track_output)
		else:
			return self.run_on_alloc(cmd=cmd, wait=wait, add_opts=add_opts, wd=wd, track_output=track_output)


	#TODO now that the asynch problem. is fixed consider moving this function to the base class
	def _run_asynch_command(self, cmd, track_output=True):
		"""
		submits a job without waiting for it.
		Throws an exception If there is currently an asynch job.
		It is recommended that this function is called from the Run method if wait=False
		returns a tuple (message,err) indicating success or failure of the submission
		"""
		try:
			popen_obj = Popen(cmd, stdout=PIPE, stderr=PIPE)
		except OSError as err:
			logger.debug(err)
			return ("","Unable to run the command. See the Error log!")
		if track_output:
			# this will also mark the process as active
			self._pending_process_info.set_process_ref(popen_obj)
			atexit.register(self.terminate_asynch_job, False, False)
		return("job submitted to the queue", "")


	def _run_synch_command(self, cmd, err_message, verbose=False):
		"""
		runs a command, waits for completion and returns the output and error.
		"""
		out = Launcher.execute_cmd(cmd_list=cmd, err_message=err_message)
		if verbose:
			logger.info(out[0])
			logger.error(out[1])
		return out

	def run_on_alloc(self, cmd, nodes=None, ppn=None, duration="", wait=True, add_opts=None,
                    partition=None, wd="", track_output=True):
		""" Runs a command on an allocation.
			The user is expected to have made an allocation
			before calling this function. On slurm we are using srun for this purpose!
			throws badOpsException is the allocation is not made
			For instance the user may have reserved 4 nodes and 10 tasks
			she can submit aprun/mpirun commands and use a parts of her reservation for each command
			param cmd list(string): commands to be executed in the form of a list of string
			param duration (string): duration in the form hh:mm:ss
			:param add_opts list(string) additional options to be added to srun
		"""
		if add_opts is None:
			add_opts = []
		if nodes is None:
			nodes = self.def_nodes
		if ppn is None:
			ppn = self.def_ppn
		if self.alloc_id is None:
			self.get_alloc(nodes, ppn, duration=duration, debug=False)
		if partition is None:
			partition = self.alloc_partition
		ntasks = ppn * nodes

		srun = ["srun", "--jobid", self.alloc_id,
						"--nodes", str(nodes),
						"--ntasks", str(ntasks)]
		if duration:
			srun += ["-t", duration]

		# I am letting the default partition to be None
		if partition is not None:
			srun += ["--partition", partition]

		if len(add_opts) > 0:
			srun.append(add_opts)
		file_name = "gen_srun_%d_%d"  % (self.instance_counter,self._get_sc_number())

		if wd:
			prepend_cd(wd, cmd)

		SlurmLauncher._put_in_bash_file(cmd, file_name)
		srun += ["bash", file_name]

		if wait:
			return self._run_synch_command(cmd=srun,
                                   		   err_message=Launcher.run_err_mess + seq_to_str(srun))
		else:
			if track_output:
				self.checkLegal_asynch()
			return self._run_asynch_command(srun, track_output=track_output)


	@staticmethod
	def _put_in_bash_file(cmd, name):
		with open(name, 'w') as destFile:
			for line in cmd:
				destFile.write("%s\n" % line)

	def run_script(self, cmd=None, wait=True, add_opts=None, wd="", clear_previous=False, track_output=True):
		# the user can clear the reference to the previous script
		if cmd is None:
			cmd = []
		if add_opts is None:
			add_opts = []
		if clear_previous:
			if not cmd:
				raise LauncherArgException("No command Specified and No script exists")
			self.script_info.clear_sc_info()

		if not self.script_info.sc_name or not os.path.isfile(self.script_info.get_sc_full_path()):
			if not cmd:
				logger.warning("No Script Was Made In Advance! Using the Default Configurations")
			if wd:
				prepend_cd(wd, cmd)
			self.make_script(cmd=cmd, add_opts=add_opts)

		elif cmd:
			if wd:
				prepend_cd(wd, cmd)
			self.add_commands_to_script(cmd)
		# Adding the -W so that squeue waits for the job to terminate
		squeue_cmd = [self._script_submit_command,  "-W", self.script_info.get_sc_full_path()]
		if add_opts:
			squeue_cmd += add_opts

		# use the launcher's static execute method to run and get the output
		if wait:
			self._run_synch_command(cmd=squeue_cmd, err_message = Launcher.run_err_mess + seq_to_str(squeue_cmd))
			#@TODO find a better solution
			time.sleep(0.5)
			return get_output_err(self.script_info)
		else:
			if track_output:
				self.checkLegal_asynch()
				self._pending_script = True
				self._pending_process_info.set_process_files(self.script_info)

			return self._run_asynch_command(squeue_cmd)

	def _write_script(self, cmd, nodes, ppn, duration, partition, add_opts, env_vars):
		"""
		Translating the arguments to a slurm script
		:param
		"""
		script_file  = self.script_info.get_sc_full_path()
		with open(script_file, "w") as dest_file:
			dest_file.write("#!/bin/bash\n\n")
			dest_file.write("#SBATCH --job-name=\"%s\"\n" % self.script_info.sc_name)
			if partition:
				dest_file.write("#SBATCH --partition=%s\n" % partition)

			dest_file.write("#SBATCH --output=%s\n" % self.script_info.output_full)
			dest_file.write("#SBATCH --error=%s\n" % self.script_info.err_full)
			dest_file.write("#SBATCH -N %s\n" % nodes)
			dest_file.write("#SBATCH -n %s\n" % str(int(ppn) * int(nodes)))
			dest_file.write("#SBATCH --time=%s\n" % duration)

			if add_opts:
				for line in add_opts:
					if not line.startswith("#SBATCH"):
						dest_file.write("#SBATCH " + line + "\n")
					else:
						dest_file.write(line + "\n")

			# write environment variables
			# taken in as list of key, value tuples
			if env_vars:
				if not isinstance(env_vars, dict):
					# TODO raise an error here
					raise Exception
				for k, v in env_vars.items():
					dest_file.write("export " + str(k) + "=" + str(v) + "\n")

			if cmd:
				for line in cmd:
					dest_file.write(line + "\n")

	def terminate_asynch_job(self, verbose=False, return_output=True):
		if not self._pending_process_info._is_active:
			return ("", "No pending process found; safe to exit")
		atexit.unregister(self.terminate_asynch_job)
		output =  self._pending_process_info.process_ref.communicate()
		if verbose:
			logger.info(output[0])
			logger.warning(output[1])

		#if the pending process was a script submission, the output and error are found in a file
		#rather than stdout
		if self._pending_script:
			self._pending_script = False
			if return_output:
				output = get_output_err(None, self._pending_process_info.get_file_tuple())

		self._pending_process_info.clear_process()
		if not return_output:
			return None
		return output
