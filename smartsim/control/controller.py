import sys
import logging
import subprocess
from os import listdir
from os.path import isdir

from launcher import SlurmLauncher, PBSLauncher
from helpers import get_SSHOME
from error.errors import SmartSimError, SSConfigError, SSUnsupportedError
from state import State
from ssModule import SSModule


class Controller(SSModule):

    def __init__(self, state, **kwargs):
        super().__init__(state)
        self.state.update_state("Simulation Control")
        self.init_args = kwargs
        self.execute = self.get_config(["execute"])


    """
    Controller Interface
    """
    def start(self):
        try:
            self.log("SmartSim Stage: " + self.state.get_state())
            self._sim()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def stop(self):
        raise NotImplementedError

    def restart(self):
        raise NotImplementedError

    def are_targets_finished():
        raise NotImplementedError

    def is_target_finished():
        raise NotImplementedError

    def _sim(self):
        for target in self.targets:
            tar_dir = self._get_target_path(target)
            tar_info = self._get_target_run_settings(target)
            run_dict = self._build_run_dict(tar_dir, tar_info)
            self.log("Executing Target: " + target)
            if self.launcher != None:
                self._run_with_launcher(tar_dir, run_dict)
            else:
                self._run_with_command(tar_dir, run_dict)


    def _build_run_dict(self, tar_dir, tar_info):
        """Build a dictionary that will be used to run with the make_script
           interface of the poseidon launcher."""

        run_dict = {}
        try:
            # search for required arguments
            self.exe = self._check_value("executable", tar_info, none_ok=False)
            self.run_command = self._check_value("run_command", tar_info, none_ok=False)
            self.launcher = self._check_value("launcher", tar_info)

            run_dict["nodes"] = self._check_value("nodes", tar_info)
            run_dict["ppn"] = self._check_value("ppn", tar_info)
            run_dict["duration"] = self._check_value("duration", tar_info)
            run_dict["partition"] = self._check_value("partition", tar_info)
            run_dict["cmd"] = self._build_run_command(tar_info)

            return run_dict
        except KeyError as e:
            raise SSConfigError("Simulation Control",
                                "Missing field under execute table in simulation.toml: " +
                                e.args[0])


    def _build_run_command(self, tar_info):
        """run_command + run_args + executable + exe_args"""
        run_args = ""
        exe_args = ""
        # init args > target_specific_args > [execute] top level args
        if "run_args" in self.execute.keys():
            run_args = self.execute["run_args"]
        if "run_args" in tar_info.keys():
            run_args = tar_info["run_args"]
        if "run_args" in self.init_args.keys():
            run_args = self.init_args["run_args"]
        if "exe_args" in self.execute.keys():
            exe_args = self.execute["exe_args"]
        if "exe_args" in tar_info.keys():
            exe_args = tar_info["exe_args"]
        if "exe_args" in self.init_args.keys():
            exe_args = self.init_args["exe_args"]

        cmd = " ".join((self.run_command, run_args, self.exe, exe_args))
        return [cmd]


    def _check_value(self, arg, tar_info, none_ok=True):
        config_value = self._check_dict(self.execute, arg)
        init_value = self._check_dict(self.init_args, arg)
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
        base_path = "".join((get_SSHOME(), self.get_config(["model","name"])))
        exp_name = self.get_config(["model", "experiment_name"])
        target_dir_path = "/".join((base_path, exp_name, target))
        return target_dir_path


    def _get_target_run_settings(self, target):
        try:
            tar_info = self.execute[target]
            return tar_info
        except KeyError:
            tar_info = dict()
            return tar_info


    def _run_with_launcher(self, tar_dir, run_dict):
        """Run with a specific type of launcher"""
        if self.launcher == "slurm":
            self._run_with_slurm(tar_dir, run_dict)
        elif self.launcher == "pbs":
            self._run_with_pbs(tar_dir, run_dict)
        elif self.launcher == "urika":
            self._run_with_urika(tar_dir, run_dict)
        else:
            raise SSUnsupportedError("Simulation Control",
                                "Launcher type not supported: "
                                + launcher)

    def _run_with_slurm(self, tar_dir, run_dict):
        launcher = SlurmLauncher.SlurmLauncher()
        for model in listdir(tar_dir):
            model_dir = "/".join((tar_dir, model))
            run_dict["dir"] = model_dir
            run_dict["output_file"] = "/".join((model_dir, model + ".out"))
            run_dict["err_file"] = "/".join((model_dir, model + ".err"))
            run_dict["clear_previous"] = True
            launcher.make_script(**run_dict)
            self.log("Running Model:  " + model)
            pid = launcher.submit_and_forget()
            self.log("Process id for " + model + " is " + str(pid))

    def _run_with_urika(self,tar_dir, run_dict):
        raise NotImplementedError

    def _run_with_pbs(self,tar_dir, run_dict):
        raise NotImplementedError


    def _run_with_command(self,tar_dir, run_dict):
        cmd = run_dict["cmd"]
        for model in listdir(tar_dir):
            self.log("Running Model:  " + model)
            model_dir = "/".join((tar_dir, model))
            run_model = subprocess.Popen(cmd, cwd=model_dir, shell=True)
            run_model.wait()




