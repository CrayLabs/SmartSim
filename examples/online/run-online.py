from smartsim import State, Orchestrator, Controller, SmartSimNode

# Create MLP regressor target for training
state = State(experiment="online-training")
state.create_target("training")
state.create_model("training-node", "training",
                   path="/lus/snx11254/spartee/smart-sim/examples/online")

# create simulation model target
state.create_target("simulation")
state.create_model("sim-model", "simulation",
                   path="/lus/snx11254/spartee/smart-sim/examples/online")

# create orchestrator
orc = Orchestrator(state)
orc.orchestrate()

# start training node
cont_dict = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "node.py",
    "run_command": "srun python"
}
train_control = Controller(state, **cont_dict)
train_control.start(target="training")

sim_dict = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "simulation.py",
    "run_command": "srun python"
}
sim_control = Controller(state, **sim_dict)
sim_control.start(target="simulation")


