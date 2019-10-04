
from smartsim import Controller, Generator, State

state = State(experiment="online-training")
state.create_target("training")
state.create_model("MLPRegressor", "training",
                   path="/lus/snx11254/spartee/smart-sim/examples/online")

cont_dict = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "node.py",
    "run_command": "srun python"
}
control = Controller(state, **cont_dict)
control.start()