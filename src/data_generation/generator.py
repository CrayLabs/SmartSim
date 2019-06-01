import shutil
import glob
import sys
import subprocess
import itertools
import toml
from os import mkdir, getcwd
from data_generation.model import NumModel


class Generator():
    """Data generation phase of the MPO pipeline. Holds internal configuration
       data that is created during the data generation stage."""

    def __init__(self, state):
        self.state = state
        self.state.update_state("Data Generation")
        self.low_models = []
        self.high_models = []

    def generate(self):
        print("MPO Stage: ", self.state.current_state)
        self.create_models()
        self.duplicate_base_configs()


    def create_models(self):
        """Populates instances of NumModel class for low and high resolution.
           obtains parameter permutations from state.

           Returns: list of high and low resolution Model objects for data
                    generation
        """

        param_dict = self.state.get_config("parameters")
        permutations = list(itertools.product(*param_dict.values()))
        params = list(param_dict.keys())

        for p in permutations:
            model_params = dict(zip(params, list(p)))
            settings = self.state.get_config("low")
            m = NumModel(model_params, settings)
            self.low_models.append(m)

        for p in permutations:
            model_params = dict(zip(params, list(p)))
            settings = self.state.get_config("high")
            m = NumModel(model_params, settings)
            self.high_models.append(m)



    def duplicate_base_configs(self):
        # TODO catch if base config path is not found
        base_path = self.state.get_config("MPO_settings")["base_config_path"]

        # Make data directories
        # TODO catch if data dirs exist already
        low_dir = getcwd() + "/../low-res-models"
        high_dir = getcwd() + "/../high-res-models"
        mkdir(low_dir)
        mkdir(high_dir)

        # Add paths to data directories to state config
        self.state.add_config("high-data-dir", high_dir)
        self.state.add_config("low-data-dir", low_dir)

        for low_run in self.low_models:
            create_low_dirs = subprocess.Popen("cp -r " + base_path +
                                                " " + low_dir + "/"+ low_run.name,
                                               shell=True)
            create_low_dirs.wait()

        for high_run in self.high_models:
            create_high_dirs = subprocess.Popen("cp -r " + base_path +
                                                " " + high_dir + "/"+ high_run.name,
                                                shell=True)
            create_high_dirs.wait()



    def run_model(self, model_to_run):
        """Runs a single configuration of a numerical model.

           Args
              model_to_run (NumModel): the model being run on slurm
        """
        # TODO fix this and make so that generator has all run settings
        executable = self.state.get_config("executable")
        num_proc = self.state.get_config("num_proc")
        run_model = subprocess.Popen("srun -n " + str(num_cores) + " " + executable,
                                     cwd="./" + model_config,
                                     shell=True)
        run_model.wait()


