import sys
import logging
import subprocess
from os import listdir
from os.path import isdir

from launcher import SlurmLauncher, PBSLauncher, UrikaLauncher
from helpers import get_SSHOME
from error.errors import SmartSimError, SSConfigError
from state import State
from ssModule import SSModule


class Controller(SSModule):

    def __init__(self, state):
        super().__init__(state)
        self.state.update_state("Simulation Control")

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


    def _sim(self):
        self._set_execute()
        for target in self.targets:
            self.log("Executing Target: " + target)
            tar_dir = self._get_target_path(target)
            tar_info = self._get_info_by_target(target)
            if self.launcher != None:
                self._run_with_launcher(tar_dir, tar_info)
            else:
                self._run_with_command(tar_dir, tar_info)


    def _set_execute(self):
        try:
            self.execute = self.get_config(["execute"])
            self.exe = self.execute["executable"]

            # set launcher or run_command
            if "launcher" in self.execute.keys():
                self.launcher == self.execute["launcher"]
                self.run_command = None
            else:
                self.run_command = self.execute["run_command"]
                self.launcher = None

        except KeyError as e:
            raise SSConfigError("Simulation Control",
                                "Missing field under execute table in simulation.toml: " +
                                e.args[0])


    def _get_info_by_target(self, target):
        try:
            tar_info = self.execute[target]
            return tar_info
        except KeyError as e:
            tar_info = {}
            return tar_info

    def _get_target_path(self, target):
        """Given a target, returns the path to the folder where that targets
           models reside"""
        base_path = "".join((get_SSHOME(), self.get_config(["model","name"])))
        exp_name = self.get_config(["model", "experiment_name"])
        target_dir_path = "/".join((base_path, exp_name, target))
        return target_dir_path



    def _run_with_launcher(self):
        pass

    def _run_with_command(self, tar_dir, tar_info):
        cmd = self._build_run_command(tar_info)
        for model in listdir(tar_dir):
            self.log("Running Model:  " + model)
            model_dir = "/".join((tar_dir, model))
            run_model = subprocess.Popen(cmd, cwd=model_dir, shell=True)
            run_model.wait()



    def _build_run_command(self, tar_info):
        """run_command + run_args + executable + exe_args"""
        run_args = ""
        exe_args = ""
        # overwrite global config with target specific config
        if "run_args" in self.execute.keys():
            run_args = self.execute["run_args"]
        if "run_args" in tar_info.keys():
            run_args = tar_info["run_args"]
        if "exe_args" in self.execute.keys():
            exe_args = self.execute["exe_args"]
        if "exe_args" in tar_info.keys():
            exe_args = tar_info["exe_args"]
        cmd = " ".join((self.run_command, run_args, self.exe, exe_args))
        return cmd

