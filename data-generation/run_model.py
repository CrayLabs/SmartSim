#!/usr/bin/python3


import sys
import subprocess


def run_model(model_config, num_cores, executable):
    """Runs a single configuration of a numerical model.

       Args
         model_config (str): name of configuration to run.

    """
    print("Running " + model_config + "...")
    run_model = subprocess.Popen("srun -n " + str(num_cores) + " " + executable,
                                 cwd="./" + model_config,
                                 shell=True)
    run_model.wait()


if __name__ == "__main__":
    run_model(sys.argv[1])
