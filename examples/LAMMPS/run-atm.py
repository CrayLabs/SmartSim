from smartsim import Experiment

experiment = Experiment("lammps_atm")

# Create ensembles
run_settings = {
    "executable": "lmp_mpi",
    "run_command": "mpirun",
    "run_args": "-np 4",
    "exe_args": "-in in.atm"
}

param_dict_1 = {"STEPS": [5, 10], "THERMO": 5}
experiment.create_ensemble("atm", params=param_dict_1, run_settings=run_settings)

base_config = "./in.atm"
experiment.generate(model_files=base_config)
experiment.start(launcher="local")

