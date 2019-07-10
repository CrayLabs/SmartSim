import sys

from os import listdir

from launcher.Launchers import SlurmLauncher
from helpers import get_SSHOME, read_config, getcwd
from error.ss_error import SmartSimError

class Controller:

    def __init__(self, state):
        self.state = state
        self.targets = self.state.get_config("targets")
        self.config = read_config(get_SSHOME() + "control.toml")


    def start_sim(self):
        try:
            self.sim()
        except SmartSimError as e:
            print(e)
            sys.exit()

    def stop_sim(self):
        raise NotImplementedError

    def restart_sim(self):
        raise NotImplementedError


    def _get_target_dir(self, target):
        ss_home = get_SSHOME()
        target_dir = "".join((ss_home, target))
        # check for the existance of the target directories
        return target_dir


    def _get_machine_info(self):
        machine = self.state.get_config("machine", self.config)
        machine_info = {}
        # if target specific info doesnt exist
        # provide defaults under machine parent table
        for target in self.targets:
            if target in machine.keys():
                machine_info[target] = machine[target]
            else:
                machine_info[target] = machine
                # TODO remove other tables
                # TODO add check for necessary fields
        return machine_info

    def sim(self):
        # simulate all models
        exe = self.state.get_config("executable_path", self.config)
        machine_info = self._get_machine_info()
        for target in self.targets:
            target_dir = self._get_target_dir(target)
            runs = listdir(target_dir)
            run_info = machine_info[target]
            for run in runs:
                path = "/".join((target_dir, run))
                self._sim(exe, run_info["nodes"], path, run_info["partition"])



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

