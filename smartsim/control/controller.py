import sys
import subprocess
import time

from os import listdir
from os.path import isdir, basename, join
from ..launcher import SlurmLauncher, PBSLauncher

from ..helpers import get_SSHOME
from ..error import SmartSimError, SSConfigError, SSUnsupportedError
from ..state import State
from ..simModule import SmartSimModule
from .job import Job

from ..utils import get_logger
logger = get_logger(__name__)


class Controller(SmartSimModule):
    """The controller module provides an interface between the numerical model
       that is the subject of Smartsim and the underlying workload manager or
       run framework. There are currently three methods of execution:

          1) Local (implemented)
          2) Slurm (implemented)
          3) PBS   (not implemented)

       :param State state: A State instance
       :param str launcher: The launcher type.  Accepted
                            options are 'local', and 'slurm'

    """

    def __init__(self, state, launcher=None, **kwargs):
        super().__init__(state, **kwargs)
        self.set_state("Simulation Control")
        self._init_launcher(launcher)
        self._jobs = []


    def start(self, target=None):
        """Start the simulations of all targets using whatever
           controller settings have been set through the Controller
           initialization or in the SmartSim configuration file.
        """
        try:
            if self.has_orchestrator():
                self._launch_orchestrator()
                self._launch_nodes()
            logger.info("SmartSim State: " + self.get_state())
            self._launch_targets(target=target)
        except SmartSimError as e:
            logger.error(e)
            raise

    def stop_all(self):
        raise NotImplementedError

    def stop(self, pid):
        raise NotImplementedError

    def get_jobs(self):
        """Return the list of jobs that this controller has spawned"""
        return self._jobs

    def get_job(self, name):
        """TODO write docs for this"""
        found = False
        for job in self._jobs:
            if job.obj.name == name:
                found = True
                return job
        if not found:
            raise SmartSimError("Job for " + name + " not found.")


    def get_job_nodes(self, job=None, wait=5):
        """Get the hostname(s) of a job from the allocation
           if no job listed, return a dictionary of jobs and nodelists

           :param Job job: A job instance
           :param int wait: time for wait before checking nodelist after job has been launched
                            defaults to 5 seconds.
           :returns: list of hostnames given a job or dict of job_name -> nodelist
        """
        if not job:
            node_dict = dict()
            time.sleep(wait)
            for job in self._jobs:
                node_dict[job.name] = self._get_job_nodes(job)
            return node_dict
        else:
            if not isinstance(job, Job):
                raise("Argument must be a Job instance")
            else:
                time.sleep(wait)
                nodes = self._get_job_nodes(job)
                return nodes


    def poll(self, interval=20, verbose=True):
        """Poll the running simulations and recieve logging
           output with the status of the job.

           :param int interval: number of seconds to wait before polling again
           :param bool verbose: set verbosity
        """
        all_finished = False
        while not all_finished:
            time.sleep(interval)
            all_finished = self.finished(verbose=verbose)

    def finished(self, verbose=True):
        """Poll all simulations and return a boolean for
           if all jobs are finished or not.

           :param bool verbose: set verbosity
           :returns: True or False for if all models have finished
        """
        # TODO make sure orchestrator doesnt effect this
        # TODO make sure NOTFOUND doesnt cause infinite loop
        statuses = []
        for job in self._jobs:
            self._check_job(job)
            statuses.append(job.status.strip())
            if verbose:
                logger.info(job)
        if "RUNNING" in statuses:
            return False
        if "assigned" in statuses:
            return False
        if "NOTFOUND" in statuses:
            return False
        return True

    def _launch_nodes(self):
        """Launch all of the SmartSimNodes declared by the user"""
        nodes = self.get_nodes()
        for node in nodes:
            # get env_vars for each connection registered by the user
            env_vars = self.state.orc.get_connection_env_vars(node.name)

            # collect and enter all run settings for Slurm
            run_dict = self._build_run_dict(node.settings)
            run_dict["wd"] = node.path
            run_dict["output_file"] = join(node.path, node.name + ".out")
            run_dict["err_file"] = join(node.path, node.name + ".err")

            # launch the job and track through job class
            self._launcher.make_script(**run_dict, env_vars=env_vars,
                                       script_name=node.name, clear_previous=True)
            pid = self._launcher.submit_and_forget(wd=node.path)
            logger.info("Launching Node: " + node.name)
            job = Job(node.name, pid, node)
            self._jobs.append(job)


    def _launch_orchestrator(self):
        """Launch the orchestrator for passing data between targets, models,
           and nodes.
        """
        settings, orc_path = self.state.orc.get_launch_settings()

        # make script and launch
        self._launcher.make_script(**settings, script_name="orchestrator", clear_previous=True)
        orc_job_id = self._launcher.submit_and_forget(wd=orc_path)

        # add orchestrator to list of jobs
        logger.info("Launching Orchestrator with pid: " + str(orc_job_id))
        orc_job = Job("orchestrator", orc_job_id, self.state.orc)
        self._jobs.append(orc_job)
        nodes = self.get_job_nodes(orc_job)[0] # only on one node for now

        # get and store the address of the orchestrator database
        self.state.orc.junction.store_db_addr(nodes, self.state.orc.port)


    def _launch_targets(self, target=None):
        """The entrypoint to simulation for multiple targets. Each target
           defines how the models wherein should be run. Each model might
           have different parameters but configurations like node count
           and ppn are determined by target.
        """
        targets = self.get_targets()
        if target:
            targets = [self.get_target(target)]
        if len(targets) < 1:
            raise SmartSimError("No targets to simulate!")
        for target in targets:
            run_dict = self._build_run_dict(target.run_settings)

            logger.info("Launching Target: " + target.name)
            if self._launcher != None:
                self._run_with_launcher(target, run_dict)
            else:
                self._run_with_command(target, run_dict)


    def _build_run_dict(self, tar_info):
        """Build a dictionary that will be used to run with the make_script
           interface of the poseidon launcher."""

        def _build_run_command(tar_dict):
            """run_command + run_args + executable + exe_args"""

            # Experiment level values required for controller to work
            exe = self.get_config("executable", aux=tar_dict, none_ok=False)
            run_command = self.get_config("run_command", aux=tar_dict, none_ok=False)

            run_args = self.get_config("run_args", aux=tar_dict, none_ok=True)
            exe_args = self.get_config("exe_args", aux=tar_dict, none_ok=True)
            if not exe_args:
                exe_args = ""
            if not run_args:
                run_args = ""

            cmd = " ".join((run_command, run_args, exe, exe_args))
            return [cmd]

        run_dict = {}
        try:
            # target level values optional because there are defaults
            run_dict["nodes"] = self.get_config("nodes", aux=tar_info, none_ok=True)
            run_dict["ppn"] = self.get_config("ppn", aux=tar_info, none_ok=True)
            run_dict["duration"] = self.get_config("duration", aux=tar_info, none_ok=True)
            run_dict["partition"] = self.get_config("partition", aux=tar_info, none_ok=True)
            run_dict["cmd"] = _build_run_command(tar_info)

            return run_dict
        except KeyError as e:
            raise SSConfigError("SmartSim could not find following required field: " +
                                e.args[0]) from e

    def _get_target_path(self, target):
        """Given a target, returns the path to the folder where that targets
           models reside"""
        target_dir_path = target.path
        if isdir(target_dir_path):
            return target_dir_path
        else:
            raise SmartSimError("Simulation target directory not found: " +
                                target)

    def _run_with_launcher(self, target, run_dict):
        """Launch all specified models with the slurm or pbs workload
           manager. job_name is the target and enumerated id.
           all output and err is logged to the directory that
           houses the model.
        """
        model_dict = target.models
        for _, model in model_dict.items():
            # get env vars for the connection of models to nodes
            env_vars = {}
            if self.has_orchestrator():
                env_vars = self.state.orc.get_connection_env_vars(model.name)

            temp_dict = run_dict.copy()
            temp_dict["wd"] = model.path
            temp_dict["output_file"] = join(model.path, model.name + ".out")
            temp_dict["err_file"] = join(model.path, model.name + ".err")
            self._launcher.make_script(**temp_dict, env_vars=env_vars,
                                       script_name=model.name, clear_previous=True)
            pid = self._launcher.submit_and_forget(wd=model.path)
            logger.debug("Process id for " + model.name + " is " + str(pid))
            job = Job(model.name, pid, model)
            self._jobs.append(job)


    def _run_with_command(self, target, run_dict):
        """Run models without a workload manager directly, instead
           using some run_command specified by the user."""
        cmd = run_dict["cmd"]
        model_dict = target.models
        for _, model in model_dict.items():
            run_model = subprocess.Popen(cmd, cwd=model.path, shell=True)
            run_model.wait()


    def _check_job(self, job):
        """Takes in job and sets job properties"""
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        status, return_code = self._launcher.get_job_stat(job_id)
        job.set_status(status)
        job.set_return_code(return_code)

    def _get_job_nodes(self, job):
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        nodes = self._launcher.get_job_nodes(job_id)
        return nodes

    def _init_launcher(self, launcher):
        """Run with a specific type of launcher"""
        if launcher is not None:
            # Init Slurm Launcher wrapper
            if launcher == "slurm":
                self._launcher = SlurmLauncher.SlurmLauncher()
            # Run all targets locally
            elif launcher == "local":
                self._launcher = None
            else:
                raise SSUnsupportedError("Launcher type not supported: "
                                        + launcher)
        else:
            raise SSConfigError("Must provide a 'launcher' argument to the Controller")
