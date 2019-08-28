from smartsim import Generator, Controller, State
from os import environ
import pytest
from os import getcwd, listdir, path
from glob import glob


def test_controller():
    
    try:
        # make sure we are on a machine with slurm
        if environ["HOST"] != "cicero":
            pytest.skip()
        else:
            # initialize State
            state= State(experiment="double_gyre", log_level="DEBUG")

            # Create targets
            quar_deg_params = {"999": [200, 400],
                            "0": [200, 400]}
            half_deg_params = {"999": [200, 400],
                            "0": [200, 400],
                            "80": 40,
                            "40": 20}
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
            
            # check if finished
            all_finished = False
            while not all_finished:
                all_finished = sim.finished(verbose=False)

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
                target_files = files.copy()
                target_files.append(target)
                target_files.append(".".join((target, ".err")))
                target_files.append(".".join((target, ".out")))
                datapath = path.join(experiment_path, target)
                all_files = [x for x in glob(datapath + "*")]
                for file in target_files:
                    if file not in all_files:
                        data_present = False
                        assert(data_present)

            assert(data_present)

    except KeyError:
        pytest.skip()                    