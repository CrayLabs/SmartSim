import sys
import logging
import subprocess
import time

from os import listdir
from os.path import isdir, basename
from launcher import SlurmLauncher, PBSLauncher

from ..helpers import get_SSHOME
from ..error import SmartSimError, SSConfigError, SSUnsupportedError
from ..state import State
from ..simModule import SmartSimModule
from .job import Job



class Controller(SmartSimModule):
    """The controller module provides an interface between the numerical model
       that is the subject of Smartsim and the underlying workload manager or
       run framework. There are currently four methods of execution:

          1) Local (implemented)
          2) Slurm (implemented)
          3) PBS   (not implemented)

       :param State state: A State instance

    """

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.set_state("Simulation Control")
        self.__set_settings()
        self._launcher = None
        self._jobs = []


    def start(self):
        """Start the simulations of all targets using whatever
           controller settings have been set through the Controller
           initialization or in the SmartSim configuration file.
        """
        try:
            self.log("SmartSim State: " + self.get_state())
            self._sim()
        except SmartSimError as e:
            self.log(e, level="error")
            raise

    def stop_all(self):
        raise NotImplementedError

    def stop(self, pid):
        raise NotImplementedError

    def get_jobs(self):
        """Return the list of jobs that this controller has spawned"""
        return self._jobs
    
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


    # TODO Make this work with jobs that dont use the launcher
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
        statuses = []
        for job in self._jobs:
            self._check_job(job)
            statuses.append(job.status)
            if verbose:
                self.log(job)
        if "RUNNING" in statuses:
            return False
        if "assigned" in statuses:
            return False
        return True

    def _sim(self):
        """The entrypoint to simulation for multiple targets. Each target
           defines how the models wherein should be run. Each model might
           have different parameters but configurations like node count
           and ppn are determined by target.
        """
        targets = self.get_targets()
        if len(targets) < 1:
            raise SmartSimError(self.get_state(), "No targets to simulate!")
        for target in targets:
            tar_info = self._get_target_run_settings(target)
            run_dict = self._build_run_dict(tar_info)

            self.log("Executing Target: " + target.name)
            if self._launcher != None:
                self._init_launcher()
                self._run_with_launcher(target, run_dict)
            else:
                self._run_with_command(target, run_dict)


    def _build_run_dict(self, tar_info):
        """Build a dictionary that will be used to run with the make_script
           interface of the poseidon launcher."""

        run_dict = {}
        try:
            # Experiment level values required for controller to work
            self._exe = self._check_value("executable", tar_info, none_ok=False)
            self._run_command = self._check_value("run_command", tar_info, none_ok=False)
            self._launcher = self._check_value("launcher", tar_info)

            # target level values optional because there are defaults
            run_dict["nodes"] = self._check_value("nodes", tar_info)
            run_dict["ppn"] = self._check_value("ppn", tar_info)
            run_dict["duration"] = self._check_value("duration", tar_info)
            run_dict["partition"] = self._check_value("partition", tar_info)
            run_dict["cmd"] = self._build_run_command(tar_info)

            return run_dict
        except KeyError as e:
            raise SSConfigError(self.get_state(),
                                "SmartSim could not find following required field: " +
                                e.args[0])


    def _build_run_command(self, tar_info):
        """run_command + run_args + executable + exe_args"""
        run_args = ""
        exe_args = ""
        # init args > target_specific_args > [execute] top level args
        # This heirarchy is explained at the module level.
        if "run_args" in self._settings.keys():
            run_args = self._settings["run_args"]
        if "run_args" in tar_info.keys():
            run_args = tar_info["run_args"]
        if "run_args" in self._init_args.keys():
            run_args = self._init_args["run_args"]
        if "exe_args" in self._settings.keys():
            exe_args = self._settings["exe_args"]
        if "exe_args" in tar_info.keys():
            exe_args = tar_info["exe_args"]
        if "exe_args" in self._init_args.keys():
            exe_args = self._init_args["exe_args"]

        cmd = " ".join((self._run_command, run_args, self._exe, exe_args))
        return [cmd]


    def _check_value(self, arg, tar_info, none_ok=True):
        """Defines the heirarchy of configuration"""
        config_value = self._check_dict(self._settings, arg)
        init_value = self._check_dict(self._init_args, arg)
        target_value = self._check_dict(tar_info, arg)
        if init_value:
            return init_value
        elif config_value:
            return config_value
        elif target_value:
            return target_value
        elif none_ok:
            return None
        else:
            raise KeyError(arg)


    def _check_dict(self, _dict, arg):
        try:
            return _dict[arg]
        except KeyError:
            return None

    def _get_target_path(self, target):
        """Given a target, returns the path to the folder where that targets
           models reside"""
        target_dir_path = target.get_target_dir()
        if isdir(target_dir_path):
            return target_dir_path
        else:
            raise SmartSimError(self.get_state(),
                                "Simulation target directory not found: " +
                                target)

    def _get_target_run_settings(self, target):
        """Retrieves the [control.<target>] table"""
        try:
            tar_info = self._settings[target.name]
            return tar_info
        except KeyError:
            tar_info = dict()
            return tar_info


    def _init_launcher(self):
        """Run with a specific type of launcher"""
        if self._launcher == "slurm":
            self._launcher = SlurmLauncher.SlurmLauncher()
        elif self._launcher == "pbs":
            self._launcher = PBSLauncher.PBSLauncher()
        else:
            raise SSUnsupportedError(self.get_state(),
                                "Launcher type not supported: "
                                + self._launcher)

    def _run_with_launcher(self, target, run_dict):
        """Launch all specified models with the slurm or pbs workload
           manager. job_name is the target and enumerated id.
           all output and err is logged to the directory that
           houses the model.
        """
        model_dict = target.get_models()
        for _, model in model_dict.items():
            temp_dict = run_dict.copy()
            temp_dict["wd"] = model.path
            temp_dict["output_file"] = "/".join((model.path, model.name + ".out"))
            temp_dict["err_file"] = "/".join((model.path, model.name + ".err"))
            self._launcher.make_script(**temp_dict, script_name=model.name, clear_previous=True)
            pid = self._launcher.submit_and_forget(wd=model.path)
            self.log("Process id for " + model.name + " is " + str(pid), level="debug")
            job = Job(model.name, pid, model)
            self._jobs.append(job)

    def _check_job(self, job):
        """Takes in job and sets job properties"""
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        status, return_code = self._launcher.get_job_stat(job_id)
        job.set_status(status)
        job.set_return_code(return_code)

    def _run_with_command(self, target, run_dict):
        """Run models without a workload manager directly, instead
           using some run_command specified by the user."""
        cmd = run_dict["cmd"]
        model_dict = target.get_models()
        for _, model in model_dict.items():
            run_model = subprocess.Popen(cmd, cwd=model.path, shell=True)
            run_model.wait()

    def __set_settings(self):
        settings = self.get_config(["control"], none_ok=True)
        if not settings:
            self._settings = {}
        else:
            self._settings = settings

    def _get_job_nodes(self, job):
        if not self._launcher:
            raise SmartSimError("No launcher set")
        job_id = job.get_job_id()
        nodes = self._launcher.get_job_nodes(job_id)
        return nodes