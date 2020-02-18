# import needed smartsim modules
from smartsim import Controller, Generator, State

# intialize state to conduct experiment
state = State(experiment="lammps_atm")

# Create ensembles
run_settings = {
    "executable": "lmp_mpi",
    "run_command": "mpirun",
    "run_args": "-np 4",
    "exe_args": "-in in.atm"
}

param_dict_1 = {"STEPS": [40, 45], "THERMO": 5}
state.create_ensemble("atm", params=param_dict_1, run_settings=run_settings)

# Supply the generator with necessary files to run the simulation
# and generate the specified models
base_config = "./in.atm"
generator = Generator(state, model_files=base_config)
generator.generate()

# Run the simulation models
control = Controller(state, launcher="local")
control.start()
