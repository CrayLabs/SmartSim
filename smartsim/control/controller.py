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
from ..ssModule import SSModule
from .job import Job

"""
there are three ways a user can specify arguments for the running
of simulations.


1) On the experiment level under the [execute] table

```toml
[execute]
nodes = 5        # All targets run with 5 nodes
```

2) On the target level under the target's table

```toml
[execute]
    [execute.some_target]
    run_args = "-np 6"
```

3) in the initialization of the Controller class

```python
control = Controller(run_args="-np 6", nodes=5)
```


There is a heirarchy of specification that goes as
follows:
    - initialization of the controller
    - experiment level (under [execute] table)
    - target level (under [execute.some_target] table)

the heirarchy is meant to allow for quick access without
having to write to the simulation.toml and seperately, intense
specification within the simulation.toml.

"""




class Controller(SSModule):
    """The controller module provides an interface between the numerical model
       that is the subject of Smartsim and the underlying workload manager or
       run framework. There are currently four methods of execution:

          1) Local (implemented)
          2) Slurm (implemented)
          3) PBS   (not implemented)
          4) Urika (not implemented)
    """

    def __init__(self, state, **kwargs):
        super().__init__(state)
        self.state.update_state("Simulation Control")
        self._init_args = kwargs
        self._execute = self._get_config(["execute"])
        self._launcher = None
        self._jobs = []


############################
### Controller Interface ###
############################


    def start(self):
        try:
            self.log("SmartSim Stage: " + self.state.get_state())
            self._sim()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def stop_all(self):
        raise NotImplementedError

    def stop(self, pid):
        raise NotImplementedError

    def poll(self, interval=20, verbose=True):
        all_finished = False
        while not all_finished:
            time.sleep(interval)
            all_finished = self.finished(verbose=verbose)

    def finished(self, verbose=True):
        statuses = []
        for job in self._jobs:
            self._check_job(job)
            statuses.append(job.status)
            if verbose:
                self.log(job)
        if "RUNNING" in statuses:
            return False
        return True




##########################

    def _sim(self):
        """The entrypoint to simulation for mutliple targets. Each target
           defines how the models wherein should be run. Each model might
           have different parameters but configurations like node count
           and ppn are determined by target.
        """
        for target in self.targets:
            tar_dir = self._get_target_path(target)
            tar_info = self._get_target_run_settings(target)
            run_dict = self._build_run_dict(tar_dir, tar_info)
            self.log("Executing Target: " + target)
            if self._launcher != None:
                self._init_launcher(tar_dir, run_dict)
            else:
                self._run_with_command(tar_dir, run_dict)


    def _build_run_dict(self, tar_dir, tar_info):
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
            raise SSConfigError(self.state.get_state(),
                                "Missing field under execute table in simulation.toml: " +
                                e.args[0])


    def _build_run_command(self, tar_info):
        """run_command + run_args + executable + exe_args"""
        run_args = ""
        exe_args = ""
        # init args > target_specific_args > [execute] top level args
        # This heirarchy is explained at the module level.
        if "run_args" in self._execute.keys():
            run_args = self._execute["run_args"]
        if "run_args" in tar_info.keys():
            run_args = tar_info["run_args"]
        if "run_args" in self._init_args.keys():
            run_args = self._init_args["run_args"]
        if "exe_args" in self._execute.keys():
            exe_args = self._execute["exe_args"]
        if "exe_args" in tar_info.keys():
            exe_args = tar_info["exe_args"]
        if "exe_args" in self._init_args.keys():
            exe_args = self._init_args["exe_args"]

        cmd = " ".join((self._run_command, run_args, self._exe, exe_args))
        return [cmd]


    def _check_value(self, arg, tar_info, none_ok=True):
        """Defines the heirarchy of configuration"""
        config_value = self._check_dict(self._execute, arg)
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
        base_path = "".join((get_SSHOME(), self._get_config(["model","name"])))
        exp_name = self._get_config(["model", "experiment"])
        target_dir_path = "/".join((base_path, exp_name, target))
        if isdir(target_dir_path):
            return target_dir_path
        else:
            raise SmartSimError(self.state.get_state(),
                                "Simulation target directory not found: " +
                                target)


    def _get_target_run_settings(self, target):
        """Retrieves the [execute.<target>] table"""
        try:
            tar_info = self._execute[target]
            return tar_info
        except KeyError:
            tar_info = dict()
            return tar_info


    def _init_launcher(self, tar_dir, run_dict):
        """Run with a specific type of launcher"""
        if self._launcher == "slurm":
            self._launcher = SlurmLauncher.SlurmLauncher()
            self._run_with_launcher(tar_dir, run_dict)
        elif self._launcher == "pbs":
            self._launcher = PBSLauncher.PBSLauncher()
            self._run_with_launcher(tar_dir, run_dict)
        else:
            raise SSUnsupportedError(self.state.get_state(),
                                "Launcher type not supported: "
                                + self._launcher)

    def _run_with_launcher(self, tar_dir, run_dict):
        """Launch all specified models with the slurm or pbs workload
           manager. job_name is the target and enumerated id.
           all output and err is logged to the directory that
           houses the model.
        """
        target = basename(tar_dir)
        for model_id, model in enumerate(listdir(tar_dir)):
            temp_dict = run_dict.copy()
            model_dir = "/".join((tar_dir, model))
            job_id = "_".join((target, str(model_id)))
            temp_dict["wd"] = model_dir
            temp_dict["output_file"] = "/".join((model_dir, model + ".out"))
            temp_dict["err_file"] = "/".join((model_dir, model + ".err"))
            temp_dict["clear_previous"] = True
            self._launcher.make_script(**temp_dict, script_name=job_id)
            pid = self._launcher.submit_and_forget(wd=model_dir)
            self.log("Process id for " + model + " is " + str(pid), level="debug")
            job = Job(job_id, pid, model_dir, model)
            self._jobs.append(job)

    def _check_job(self, job):
        """Takes in job and sets job properties"""
        status, return_code = self._launcher.get_job_stat(job.jid)
        job.set_status(status)
        job.set_return_code(return_code)


    def _run_with_command(self,tar_dir, run_dict):
        """Run models without a workload manager using
           some run_command specified by the user."""
        cmd = run_dict["cmd"]
        for model in listdir(tar_dir):
            self.log("Running Model:  " + model)
            model_dir = "/".join((tar_dir, model))
            run_model = subprocess.Popen(cmd, cwd=model_dir, shell=True)
            run_model.wait()




