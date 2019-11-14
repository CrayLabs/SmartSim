from smartsim import State, Controller

state = State(experiment="online-training")
train_settings = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "regressor.py",
    "run_command": "srun python"
}
state.create_node("training-node", **train_settings)
state.create_model("sim-model",
                   path="/lus/snx11254/spartee/smart-sim/examples/online")

state.create_orchestrator()
state.register_connection("sim-model", "training-node")

sim_dict = {
    "launcher": "slurm",
    "nodes": 1,
    "duration": "2:00:00",
    "executable": "simulation.py",
    "run_command": "srun python"
}
sim_control = Controller(state, **sim_dict)
sim_control.start()
sim_control.poll()


