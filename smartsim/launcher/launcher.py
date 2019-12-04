"""
Interface for implementing unique launchers on distributed
systems.
	- Examples: Slurm, pbs pro, Urika-xc, etc

Job Execution Interface:
	- run(args) -> recommended to default to script execution mode
	- run_script(args) -> scripts can be made (make_script(args) or imported (import_script(path))
	- run_on_alloc(args) -> get_alloc(args) makes an allocation
	- submit_and_forget(command) -> submits async and doesn't track execution

All of these methods keep track if the output and error in some shape and form
(e.g keeping a reference to the spawned process, recording the output and error file, etc)
except for submit_and_forget.
"""

import abc
from subprocess import PIPE, Popen, CalledProcessError
from .launcherUtil import LauncherArgException, ScriptInfo
from .launcherUtil import LauncherBadOpException, get_output_err, seq_to_str, prepend_cd
import os
import time
import atexit

from ..utils import get_logger
logger = get_logger(__name__)


class Launcher(abc.ABC):
	static_counter = 0
	cancel_err_mess = "Unable to revoke your allocation for jobid %s\n"
	run_err_mess = "An error occurred while trying to run the command:\n"

	def __init__(self, def_nodes=2, def_ppn=1, def_partition=None, def_queue=None, def_duration="30:00"):
		""" :param def_nodes: Default number of nodes to allocation
			:param def_ppn: Default processes per node
			:param def_partition: Default partition to select
			:param def_queue: Default queue
			:param def_duration: is the default walltime HH:MM:SS
			:param sc_name:  the name of generated script"""
		Launcher.static_counter += 1
		self.instance_counter = Launcher.static_counter
		self._script_submit_command = ""
		self.def_nodes = def_nodes
		self.def_ppn = def_ppn
		self.def_partition = def_partition
		self.def_queue = def_queue
		self.alloc_id = None
		self.def_duration = def_duration
		self._pending_process_info = self.PendingProcessInfo()
		self.script_info = ScriptInfo("","", "", "")
		self.gen_script_count = 0
		self.alloc_ids = set()
		atexit.register(self.clean_up)
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
	def _get_alloc_cmd(self, nodes, ppn, partition, start_time, duration, add_opts, to_string=False, debug=False):
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
	def run(self, cmd=None, use_script=True, wait=True, add_opts=None, wd=""):
		"""This is used to implement the default launch behavior with minimum adjustment by the user
		For better control, the user should use use either
		launcher.make_script(args) and launcher.run_script(args) or
		launcher.get_alloc(args) launcher.run_on_alloc(args)
		:param cmd: Command to run in the form of a list of strings
		:param nodes: Override the number of nodes to run the command on
		:param ppn: Override the number of processes per node to run the command on
		:param add_opts: Any additional options to pass to the run command
		:param wd (str) specifies if the job needs to be run from a specific dir
		:return:
		"""
		pass

	@abc.abstractmethod
	def run_on_alloc(self, cmd, nodes=None, ppn=None, duration="", wait=True, add_opts=None, partition=None, wd=""):
		"""
		Runs a command on an allocation.
		The user is expected to have made an allocation before calling this function
		throws badOpsException is the allocation is not made
		For instance the user may have reserved 4 nodes and 10 tasks
		she can submit aprun/mpirun commands and use a parts of her reservation for each command
		"""
		pass

	@abc.abstractmethod
	def run_script(self, cmd=None, wait=True, add_opts=None, wd="", clear_previous=False):
		"""
		:param cmd list(string): the commands to be written/appended to the script(see explanation below)
		:param wait bool: whether to run the script synchronously or asynchronously
		:param add_opts list(string): the additional launcher options to be written at the top of file
                                     (ignored if script exists; see below)
		In case the launcher has a reference to a previous script, the same one will be used and
		the commands in cmd (if any) will be appended to it. To make a new script, either use make_script
		or use Launcher.scriptinfo.clear_sc_info()
		"""
		pass

	@abc.abstractmethod
	def _get_free_cmd(self, alloc_id):
		"""
		returns the workload manager dependant command to release the allocation
		This is used in the free_alloc function which is implemented below.
		This method is meant to be used privately
		:param alloc_id (string): The allocation id to be released.
		"""
		pass

	@abc.abstractmethod
	def _write_script(self, cmd, nodes, ppn, duration, partition, add_opts):
		"""
		a private method that translates the argument to proper directives
		based on the underlying workload manager
		some of these arguments may not be relevant for some launchers
		for instance, PBS does not have partitions so it should be find to leave them None
		"""
		pass


	@abc.abstractmethod
	def terminate_asynch_job(self, verbose=False, return_output=True):
		"""
		Terminates the pending process that the launcher is managing.
		If there is currently no pending process, an error message is returned
		:param verbose (bool): indicates whether or not the error and output
		:param return_output (bool): indicating whether or not the stdout/stderr associated
		with this process should be read and returned
		generated by communicating with the process should be printed
		"""
		pass

	#--------------------------- General methods with implementations -------------------

	def set_alloc_id(self, alloc_id):
		self.alloc_id = str(alloc_id)

	def has_pending_process(self):
		if self._pending_process_info._is_active:
			return True
		return False

	def poll_pending_process(self):
		"""
		A function to be used by the user to check if a pending process has completed
		it the pending process has terminated return the return code otherwise None.
		It's up to the user to clear the pending process and gets its output by calling terminate
		"""
		if not self._pending_process_info._is_active:
			raise LauncherBadOpException("No pending process")
		return self._pending_process_info.process_ref.poll()


	def launch_urika(self, cmd, nodes=None, urika_opts=None,
					start_analytics=True, ppn=None, wait=True, duration="", wd="", module_load=False):
		"""
		launches a urika command using run_training or start_analytics utilizing the underlying workload
		manager.

		:param cmd (str): the command to be run inside the container
		:param on_alloc (bool): if true, the command will be run on the allocation otherwise,
		 an script is made and submitted
		:param module_load (bool): whether or not module load analytics should be executed first
		"""
		wd = wd.strip()
		if urika_opts is None:
			urika_opts = []
		urika_cmd = []
		if module_load:
			urika_cmd.append("module load analytics")
		if wd:
			urika_cmd.append("cd %s" % wd)
		if start_analytics:
			analytics_str = "start_analytics "
			if urika_opts:
				analytics_str += " ".join(urika_opts)
				analytics_str += " -t \"%s\"" % cmd
			urika_cmd.append(analytics_str)
		else:
			analytics_str = "run_training "
			if ppn:
				analytics_str += "--ppn %d " % ppn
			if urika_opts:
				analytics_str += " ".join(urika_opts)
				analytics_str += " \"%s\"" % cmd
			urika_cmd.append(analytics_str)


		self.make_script(cmd=urika_cmd, nodes=nodes, ppn=ppn, duration=duration, clear_previous=True)
		if wait:
			return self.run_script(wait=True)
		else:
			return self.submit_and_forget()


	def submit_and_forget(self, cmd=None, clear_previous=False, wd=None):
		"""
		generates a script and submits it without waiting for it to finish or piping its output
		if the class already tracks a script and cmd is not empty, the commands will be appended to it
		to discard the previous generated script and start anew, use self.script_info.clear_sc_info()
		:param cmd list(string) each string in cmd becomes a line in the script
		:return (str) the job id of the submission "-1" otherwise
		"""
		if clear_previous:
			self.script_info.clear_sc_info()

		if not self.script_info.sc_name or not os.path.isfile(self.script_info.get_sc_full_path()):
			if not cmd:
				raise LauncherArgException("No command specified and no script exists")

			logger.warning("No script was made in advance. using the default configurations")
			self.make_script(cmd=cmd)

		elif cmd:
			self.add_commands_to_script(cmd)
		qsub_cmd = [self._script_submit_command, self.script_info.get_sc_full_path()]
		out,_ =  Launcher.execute_cmd(qsub_cmd, err_message = Launcher.run_err_mess + seq_to_str(qsub_cmd), cwd=wd)
		if not out:
			return "-1"
		out = out.strip().split()
		return out[-1]


	def free_alloc(self, alloc_id=None):
		"""
		if there is an active researvation, or if alloc_id is specified it gets cancelled
		"""
		if alloc_id is None:
			if self.alloc_id is None:
				logger.info("No allocation found, safe to exit")
				return
			else:
				alloc_id = self.alloc_id

		(cancel_cmd, cancel_err_mess) = self._get_free_cmd(alloc_id)
		try:
			_, err = Launcher.execute_cmd(cancel_cmd, err_message=cancel_err_mess)

		except CalledProcessError:
			logger.debug("Unable to revoke your allocation for jobid %s" % alloc_id)
			logger.debug("The job may have already timed out, or you may need to cancel the job manually")
			logger.debug(err)
			return

		logger.info("Successfully Freed Allocation %s" % alloc_id)
		self.alloc_id = None

	#@TODO allow user to specify the script name using user's script name
	def make_script(self, cmd=None, nodes=None, ppn=None, add_opts=None, output_file=None,
                   err_file=None,script_name=None, duration="", wd="", partition=None, debug=False,
                   env_vars=None, clear_previous=True):
		"""
		generate a customized script
		the method _write_script should be overwritten to translate the args to
		commands based on the underlying workload manager
		When we generate a script we put it in our pre-defined location
		:param cmd (List(String)) each item is a command that translates to one line of the script
		:param output_file(str) complete absolute path to the output file
		:param err_file (str) complete absolute path to the error file
		:param script_file (str) name of the script to submit
		"""
		if clear_previous:
			self.script_info.clear_sc_info()
		if not cmd:
			raise LauncherArgException("The command has not been specified!")
		if debug:
			logger.debug("This is the command that will be in the script:\n%s" % cmd)

		if script_name:
			self.script_info.sc_name = script_name

		if not self.script_info.sc_name:
			# use a default name and tag it with instance number and script number
			self.script_info.sc_name = "gen_job_sc_%d_%d"  % (self.instance_counter, self._get_sc_number())
			self.script_info.extention = ".sh"

		if wd:
			self.script_info.sc_path = wd

		if nodes is None:
			nodes = self.def_nodes
		if ppn is None:
			ppn = self.def_ppn
		if partition is None:
			partition = self.def_partition
		#handling error and and output files
		if output_file is None:
			self.script_info.output_full = self.script_info.sc_path + self.script_info.sc_name + "_output.txt"
		else:
			self.script_info.output_full = output_file
		if err_file is None:
			self.script_info.err_full = self.script_info.sc_path + self.script_info.sc_name + "_err.txt"
		else:
			self.script_info.err_full = err_file
		if not duration:
			duration = self.def_duration


		self._write_script(cmd, nodes, ppn, duration, partition, add_opts, env_vars)


	def add_commands_to_script(self, cmd, debug=False):
		"""
		cmd: List(String): the commands to be appended to the script file
		Just in case during the program operation a command needs to be added to the script
		The function treats each string in cmd as a string and appends it to the script file
		"""
		if not os.path.isfile(self.script_info.get_sc_full_path()):
			raise LauncherBadOpException("You need to create a script first before adding commands to it")
		with open (self.script_info.get_sc_full_path(), "a") as sc_file:
			for cmd in cmd:
				sc_file.write(cmd + "\n")
		if debug:
			logger.debug("added commands to " + self.script_info.get_sc_full_path())


	def clean_up(self):
		"frees allocations and terminates the background job (if any)"
		try:
			for alloc_id in self.alloc_ids:
				self.free_alloc(alloc_id)
				self.alloc_ids.clear()
		except Exception as ex:
			logger.warning(ex)

		try:
			self.terminate_asynch_job()
		except Exception as ex:
			logger.warning(ex)

	def checkLegal_asynch(self):
		"""
		Since we can only have one asynch job running in the background
		this method checks and raises an exception if this object is already
		tracking a bg job
		"""
		if self._pending_process_info._is_active:
			raise LauncherBadOpException("Another process is already executing")

	def _get_sc_number(self):
		"""
		internal private method used for generting unique script names
		"""
		sc_no = self.gen_script_count
		self.gen_script_count += 1
		return sc_no

#-------------------- General static methods with implementations -------------------
	@staticmethod
	def execute_cmd(cmd_list, err_message="", debug=False, shell=False, cwd=None, verbose=False):
		"""
			This is the function that runs a shell command and returns the output
			It will raise exceptions if the commands did not run successfully
		"""
		if verbose:
			print("Executing shell command: %s" % " ".join(cmd_list))
		# spawning the subprocess and connecting to its output
		proc = Popen(cmd_list, stdout=PIPE, stderr=PIPE, shell=shell, cwd=cwd)
		try:
			# waiting for the process to terminate and capture its output
			out, err = proc.communicate()
		except CalledProcessError as e:
			logger.error("Exception while attempting to start a shell process")
			raise e

		if proc.returncode is not 0:
			logger.error("Command \"%s\" returned non-zero" % " ".join(cmd_list))
			logger.error(err)
			logger.error(err_message)
			# raise exception removed: no need to throw an exception here!

		# decoding the output and err and return as a string tuple
		return (out.decode('utf-8'), err.decode('utf-8'))

	class PendingProcessInfo:
		"""
		an inner class so that we can handle a pending procees and still submit jobs synchronously
		example would be to populate fields in the run_asynch method and clear in terminate and clean_up functions
		it has replaced the self._pending_process instance attribute
		"""
		def __init__(self):
			self.out_file = None
			self.err_file = None
			self._is_active = False
			self.process_ref = None

		def set_process_files(self, scInfo):
			self.out_file = scInfo.output_full
			self.err_file = scInfo.err_full

		def set_process_ref(self, popen_obj):
			"""
			stores the referene to the Popen instance, and marks the pending process as active
			"""
			self._is_active = True
			self.process_ref = popen_obj

		def get_file_tuple(self):
			return (self.out_file, self.err_file)

		def clear_process(self):
			self.process_ref = None
			self._is_active = False
			self.out_file = self.err_file = ""
