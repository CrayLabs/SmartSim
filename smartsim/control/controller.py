import sys
import subprocess
import time

from os import listdir
from os.path import isdir, basename, join
from launcher import SlurmLauncher, PBSLauncher

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
       run framework. There are currently four methods of execution:

          1) Local (implemented)
          2) Slurm (implemented)
          3) PBS   (not implemented)

       :param State state: A State instance

    """

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        self.set_state("Simulation Control")
        self._init_launcher()
        self._jobs = []


    def start(self, target=None):
        """Start the simulations of all targets using whatever
           controller settings have been set through the Controller
           initialization or in the SmartSim configuration file.
        """
        try:
            if self.has_orcestrator():
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
            raise SmartSimError(self.get_state(),
                                "Job for " + name + " not found.")


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
        statuses = []
        for job in self._jobs:
            self._check_job(job)
            statuses.append(job.status)
            if verbose:
                logger.info(job)
        if "RUNNING" in statuses:
            return False
        if "assigned" in statuses:
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
            raise SmartSimError(self.get_state(), "No targets to simulate!")
        for target in targets:
            tar_info = self._get_target_run_settings(target)
            run_dict = self._build_run_dict(tar_info)

            logger.info("Launching Target: " + target.name)
            if self._launcher != None:
                self._run_with_launcher(target, run_dict)
            else:
                self._run_with_command(target, run_dict)


    def _build_run_dict(self, tar_info):
        """Build a dictionary that will be used to run with the make_script
           interface of the poseidon launcher."""

        def _build_run_command(tar_dict, settings):
            """run_command + run_args + executable + exe_args"""
            run_args = ""
            exe_args = ""

            # Experiment level values required for controller to work
            exe = self._check_value("executable", tar_dict, settings, none_ok=False)
            run_command = self._check_value("run_command", tar_dict, settings, none_ok=False)

            # get run_args
            if "run_args" in self._init_args.keys():
                run_args = self._init_args["run_args"]
            if "run_args" in settings.keys():
                run_args = settings["run_args"]
            if "run_args" in tar_dict.keys():
                run_args = tar_dict["run_args"]

            # get exe_args
            if "exe_args" in self._init_args.keys():
                exe_args = self._init_args["exe_args"]
            if "exe_args" in settings.keys():
                exe_args = settings["exe_args"]
            if "exe_args" in tar_dict.keys():
                exe_args = tar_dict["exe_args"]


            cmd = " ".join((run_command, run_args, exe, exe_args))
            return [cmd]

        run_dict = {}
        try:
            settings = self._get_settings()

            # target level values optional because there are defaults
            run_dict["nodes"] = self._check_value("nodes", tar_info, settings)
            run_dict["ppn"] = self._check_value("ppn", tar_info, settings)
            run_dict["duration"] = self._check_value("duration", tar_info, settings)
            run_dict["partition"] = self._check_value("partition", tar_info, settings)
            run_dict["cmd"] = _build_run_command(tar_info, settings)

            return run_dict
        except KeyError as e:
            raise SSConfigError(self.get_state(),
                                "SmartSim could not find following required field: " +
                                e.args[0])


    def _check_value(self, arg, tar_info, user_settings, none_ok=True):
        """Defines the heirarchy of configuration"""

        def _check_dict(_dict, arg):
            try:
                return _dict[arg]
            except KeyError:
                return None

        config_value = _check_dict(user_settings, arg)
        init_value = _check_dict(self._init_args, arg)
        target_value = _check_dict(tar_info, arg)

        if target_value:      # under [control.target] table or node or orc
            return target_value
        elif config_value:        # under [control] table
            return config_value
        elif init_value:        # controller init
            return init_value
        elif none_ok:
            return None
        else:
            raise KeyError(arg)



    def _get_target_path(self, target):
        """Given a target, returns the path to the folder where that targets
           models reside"""
        target_dir_path = target.path
        if isdir(target_dir_path):
            return target_dir_path
        else:
            raise SmartSimError(self.get_state(),
                                "Simulation target directory not found: " +
                                target)


    def _get_target_run_settings(self, target):
        """Retrieves the [control.<target>] table"""
        settings = self._get_settings()
        try:
            tar_info = settings[target.name]
            return tar_info
        except KeyError:
            tar_info = dict()
            return tar_info

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
            if self.has_orcestrator():
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

    def _init_launcher(self):
        """Run with a specific type of launcher"""
        launcher = self.get_config(["control", "launcher"], none_ok=True)
        if launcher is not None:
            # Init Slurm Launcher wrapper
            if launcher == "slurm":
                self._launcher = SlurmLauncher.SlurmLauncher()
            # Run all targets locally
            elif launcher == "" or launcher == "local":
                self._launcher = None
            else:
                raise SSUnsupportedError(self.get_state(),
                                        "Launcher type not supported: "
                                        + launcher)
        else:
            raise SSConfigError(self.get_state(),
                                "Must provide a 'launcher' argument to the Controller")

    def _get_settings(self):
        settings = self.get_config(["control"], none_ok=True)
        if settings:
            return settings
        else:
            return dict()