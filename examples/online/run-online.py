from smartsim import State, Controller

state = State(experiment="online-training")
train_settings = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "regressor.py",
    "run_command": "srun python"
}

# Make state aware of ML model and simulation model
state.create_node("training-node", **train_settings)
state.create_model("sim-model",
                   path="/lus/snx11254/spartee/smart-sim/examples/online")

# Orchestrate the connection between the ML model and simulation
state.create_orchestrator()
state.register_connection("sim-model", "training-node")

sim_dict = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "simulation.py",
    "run_command": "srun python"
}

# Launch everything and poll Slurm for statuses
sim_control = Controller(state, **sim_dict)
sim_control.start()
sim_control.poll()


