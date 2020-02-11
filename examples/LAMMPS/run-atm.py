
# import needed smartsim modules
from smartsim import Controller, Generator, State

# intialize state to conduct experiment
state = State(experiment="lammps_atm")

# Create ensembles
param_dict_1 = {"STEPS": [20, 25],
                "THERMO": 5}
param_dict_2 = {"STEPS": [30, 40],
                "THERMO": 5}
state.create_ensemble("atm", params=param_dict_1)
state.create_ensemble("atm-2", params=param_dict_2)

# Supply the generator with necessary files to run the simulation
# and generate the specified models
base_config = "./in.atm"
GEN = Generator(state, model_files=base_config)
GEN.generate()

# Save the generated models
state.save()

# Run the simulation models
CTRL = Controller(state, launcher="local", executable="lmp_mpi", run_command="mpirun",
run_args="-np 6", exe_args="-in in.atm")
CTRL.start()

