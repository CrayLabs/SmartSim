from smartsim import State, Controller

# Create MLP regressor target for training
state = State(experiment="online-training")

# Set run settings for a training node
train_settings = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "regressor.py",
    "run_command": "srun python"
}
state.create_node("training-node", **train_settings)

# create simulation model target
state.create_target("simulation")
state.create_model("sim-model", "simulation",
                   path="/lus/snx11254/spartee/smart-sim/examples/online")

# Create the orchestrator
state.create_orchestrator()
state.register_connection("sim-model", "training-node")

# settings for the simulation model
sim_dict = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "simulation.py",
    "run_command": "srun python"
}
sim_control = Controller(state, **sim_dict)

# launch the nodes, targets, and orchestrator
sim_control.start()
sim_control.poll()


