import sys
import logging
from os import listdir
from os.path import isdir

from launcher.Launchers import SlurmLauncher
from helpers import get_SSHOME, read_config, getcwd
from error.errors import SmartSimError, SSConfigError
from state import State
from ssModule import SSModule

class Controller(SSModule):

    def __init__(self, state):
        super().__init__(state)
        self.state.update_state("Simulation")

    """
    Controller Interface
    """
    def start_sim(self):
        try:
            logging.info("SmartSim Stage: %s", self.state.get_state())
            self.sim()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def stop_sim(self):
        raise NotImplementedError

    def restart_sim(self):
        raise NotImplementedError


    def _get_target_dir(self, target):
        #TODO let user specify this directory
        ss_home = get_SSHOME()
        target_dir = "".join((ss_home, target))
        if not isdir(target_dir):
            raise SmartSimError(self.state.get_state(),
                                "Target directories not found, simulation cancelled")
        else:
            return target_dir

    def _get_target_info(self, target):
        """Checks for necessary run infomation

           Required information
             - nodes
             - parition

           Args
             target (str): The target model to return info for

           Returns
             Dict of run information for the workload manager
        """
        target_info = self.get_config(["machine", target])
        required = {"partition", "nodes"}
        if not set(target_info.keys()).issuperset(required):
            raise SSConfigError(self.state.get_state(),
                                "One of required fields for workload manager not found in config: "
                                + ", ".join(required))
        else:
            return target_info


    def sim(self):
        # simulate all models
        exe = self.get_config(["execute", "executable_path"])
        for target in self.targets:
            target_dir = self._get_target_dir(target)
            target_info = self._get_target_info(target)
            runs = listdir(target_dir)

            for run in runs:
                path = "/".join((target_dir, run))
                self._sim(exe, target_info["nodes"], path, target_info["partition"])



    def _sim(self, exe, nodes, model_path, partition="iv24"):
        """Simulate a model that has been configured by the generator
           Currently uses the slurm launcher

           Args
              exe        (str): path to the compiled numerical model executable
              nodes      (int): number of nodes to run on for this model
              model_path (str): path to dir that houses model configurations
              partition  (str): type of proc to run on (optional)
        """
        launcher = SlurmLauncher(def_nodes=nodes, def_partition=partition)
        launcher.validate()
        launcher.get_alloc()
        launcher.run([exe], cwd=model_path)
        launcher.free_alloc()

