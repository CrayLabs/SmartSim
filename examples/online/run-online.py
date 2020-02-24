from smartsim import State, Controller

state = State(experiment="online-training")
train_settings = {
    "nodes": 1,
    "duration": "1:00:00",
    "executable": "python regressor.py",
}
sim_settings = {
    "nodes": 1,
    "duration": "1:00:00",
    "executable": "python simulation.py",
}
# Make state aware of ML model and simulation model
state.create_node("training-node", run_settings=train_settings)
state.create_ensemble("sim-ensemble", run_settings=sim_settings)
state.create_model("sim-model", "sim-ensemble")

# Orchestrate the connection between the ML model and simulation
orc = state.create_orchestrator(cluster_size=3)
state.register_connection("sim-model", "training-node")


# Launch everything and poll Slurm for statuses
sim_control = Controller(state, launcher="slurm")
sim_control.start()
sim_control.poll()
sim_control.release()


