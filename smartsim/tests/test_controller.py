import pytest

from os import getcwd, listdir, path, environ
from glob import glob
from shutil import rmtree

from smartsim import Generator, Controller, State


def test_controller():

    try:
        # make sure we are on a machine with slurm
        if environ["HOST"] != "cicero":
            pytest.skip()
        else:

            smartsim_home = environ["SMARTSIMHOME"]
            experiment_dir = path.join(smartsim_home, "double_gyre")
            # Cleanup from previous test
            if path.isdir(experiment_dir):
                rmtree(experiment_dir)

            # initialize State
            state= State(experiment="double_gyre", log_level="DEBUG")

            # Create targets
            quar_deg_params = {"999": [200, 400],
                               "0": [200, 400],
                               "120": 1}
            half_deg_params = {"999": [200, 400],
                               "0": [200, 400],
                               "80": 40,
                               "40": 20,
                               "120": 1}
            state.create_target("quar-deg", params=quar_deg_params)
            state.create_target("half-deg", params=half_deg_params)



            # Generate Models
            gen = Generator(state, model_files="MOM6/MOM6_base_config")
            gen.generate()

            control_dict = {"nodes":2,
                            "executable":"MOM6",
                            "run_command":"srun",
                            "launcher": "slurm",
                            "partition": "iv24"}
            sim = Controller(state, **control_dict)
            sim.start()

            while(sim.poll(verbose=False)):
               continue

            # check if all the data is there
            # files to check for
            #     input.nml            (model config, make sure generator is copying)
            #     <target_name>        (for script from launcher)
            #     <target_name>.err    (for err files)
            #     <target_name>.out    (for output)
            #     ocean_mean_month.nc  (make sure data is captured)

            data_present = True
            files = ["input.nml", "ocean_mean_month.nc"]
            experiment_path = state.get_experiment_path()
            targets = listdir(experiment_path)
            for target in targets:
                target_path = path.join(experiment_path, target)
                for model in listdir(target_path):
                    model_files = files.copy()
                    model_files.append(model)
                    model_files.append(".".join((model, "err")))
                    model_files.append(".".join((model, "out")))
                    model_path = path.join(target_path, model)
                    print(model_path)
                    all_files = [path.basename(x) for x in glob(model_path + "/*")]
                    print(all_files)
                    print(model_files)
                    for model_file in model_files:
                        if model_file not in all_files:
                            print(model_file)
                            data_present = False
                            assert(data_present)

            assert(data_present)

            # Cleanup from previous test
            if path.isdir(experiment_path):
                rmtree(experiment_path)

    except KeyError:
        pytest.skip()
