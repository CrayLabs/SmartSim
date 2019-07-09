
from launcher.Launchers import SlurmLauncher


class Controller:

    def __init__():
        # start here

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

