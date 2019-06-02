
import subprocess
from glob import glob

class ModelRunner:
    """Initialized inside of the generator class to run numerical models
       after the base configuration has been duplicated by the generator
       class and the model specific configurations have been placed into
       each duplicated model by the model specific module for a particular
       numerical model.

       Two instances of the ModelRunner class will be initialized in the
       generator class. One for the low resolution models and one for the
       high resolution models. Two instances are created because the number
       of nodes for each resolution is most likely going to be different.


    """

    def __init__(self, executable_path, nodes, proc_per_node):
        self.exe = executable_path
        self.nodes = nodes
        self.procs = proc_per_node * nodes


    def run_all_models(self, model_dir):
        self.allocate_slurm_job()
        for model in glob(model_dir):
            self.run_model(model)



    def allocate_slurm_job(self):
        """Performs a Salloc for a given set of models to be run.
           Recieves job information from mpo-config.toml"""
        command = "salloc "
        node_count = "-N " + str(self.nodes) + " "
        #TODO add option to add in time, otherwise max time on cicero
        time = "-t 23:00:00 "
        command += node_count + time

        submit_job = subprocess.Popen(command, shell=True)
        submit_job.wait()

    def run_model(self, model_to_run):
        """Runs a single configuration of a numerical model.

           Args
              model_to_run (NumModel): the model being run on slurm
        """
        run_model = subprocess.Popen("srun -n " + str(self.procs) + " " + self.exe,
                                     cwd=model_to_run,
                                     shell=True)
        run_model.wait()

