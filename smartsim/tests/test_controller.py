import pytest

from os import getcwd, listdir, path, environ, mkdir
from glob import glob
from shutil import rmtree, which

from smartsim import Generator, Controller, State


def test_controller():

    # see if we are on slurm machine
    if not which("srun"):
        pytest.skip()
    # see if MOM6 has been compiled
    elif not which("MOM6"):
        pytest.skip()
    else:

        experiment_dir = "./double_gyre"
        if path.isdir(experiment_dir):
            rmtree(experiment_dir)

        state= State(experiment="double_gyre")
        quar_deg_params = {"KH": [200, 400],
                            "KHTH": [200, 400],
                            "x_resolution": 80,
                            "y_resolution": 40,
                            "months": 1}
        half_deg_params = {"KH": [200, 400],
                            "KHTH": [200, 400],
                            "x_resolution": 40,
                            "y_resolution": 20,
                            "months": 1}
        state.create_target("quar-deg", params=quar_deg_params)
        state.create_target("half-deg", params=half_deg_params)

        gen = Generator(state, model_files="../../examples/MOM6/MOM6_base_config")
        gen.generate()

        # because it is running in the default partition
        # this works with current allocation strategy.
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
        experiment_path = sim.get_experiment_path()
        targets = listdir(experiment_path)
        for target in targets:
            target_path = path.join(experiment_path, target)
            for model in listdir(target_path):
                model_files = files.copy()
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

        # Cleanup
        sim.release()
        if path.isdir(experiment_path):
            rmtree(experiment_path)


def test_no_generator():
    """Test the controller when the model files have not been created by
       a generation strategy"""

    # see if we are on slurm machine
    if not which("srun"):
        pytest.skip()
    # see if cp2k has been compiled
    if not which("cp2k.psmp"):
        pytest.skip()

    output_file_dir = getcwd() + "/test-output"
    if path.isdir(output_file_dir):
        rmtree(output_file_dir)
    mkdir(output_file_dir)

    state = State(experiment="test_output")

    target_run_settings = {"executable": "cp2k.psmp",
                           "run_command": "srun",
                           "partition": "gpu",
                           "exe_args": "-i ../test_configs/h2o.inp",
                           "nodes": 1}
    state.create_target("test-target", run_settings=target_run_settings)
    state.create_model("test", target="test-target", path=output_file_dir)

    control = Controller(state, launcher="slurm")
    control.start()

    while(control.poll(verbose=False)):
        continue

    file_list = ["h2o-1.ener", "test.err", "test.out", "h2o-pos-1.xyz"]
    file_list = [path.join(output_file_dir, f) for f in file_list]
    for f in file_list:
        assert(path.isfile(f))

    control.release()
    if path.isdir(output_file_dir):
        rmtree(output_file_dir)



def test_target_configs():
    """Test the controller for when targets are provided their own run configurations"""

    # see if we are on slurm machine
    if not which("srun"):
        pytest.skip()
    # see if cp2k has been compiled
    if not which("cp2k.psmp"):
        pytest.skip()

    output_file_dir = getcwd() + "/target-test"
    if path.isdir(output_file_dir):
        rmtree(output_file_dir)

    state = State(experiment="target-test")
    target_params = {"executable": "cp2k.psmp",
                     "run_command": "srun",
                     "partition": "gpu",
                     "exe_args": "-i h2o.inp",
                     "nodes": 1}
    state.create_target("test-target", run_settings=target_params)
    state.create_model("test", "test-target")

    gen = Generator(state, model_files="test_configs/h2o.inp")
    gen.generate()

    sim_params = {"launcher": "slurm"}
    control = Controller(state, **sim_params)
    control.start()

    while(control.poll(verbose=False)):
        continue

    control.release()
    if path.isdir(output_file_dir):
        rmtree(output_file_dir)