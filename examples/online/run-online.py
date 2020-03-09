from smartsim import Experiment

experiment = Experiment("online-training")
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
# Make experiment aware of ML model and simulation model
experiment.create_node("training-node", run_settings=train_settings)
experiment.create_ensemble("sim-ensemble", run_settings=sim_settings)
experiment.create_model("sim-model", "sim-ensemble")

# Orchestrate the connection between the ML model and simulation
orc = experiment.create_orchestrator(cluster_size=3)
experiment.register_connection("sim-model", "training-node")


# Launch everything and poll Slurm for statuses
experiment.start(launcher="slurm")
experiment.poll()
experiment.release()


