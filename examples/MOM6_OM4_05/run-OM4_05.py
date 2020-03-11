from smartsim import Experiment

experiment = Experiment("Reconstruct_OM4_05")
OM4_05_settings= {
    "ppn":24,
    "nodes": 60,
    "duration": "0:10:00",
    "executable": "MOM6",
    "partition": "iv24"
}
reconstruct_settings = {
    "nodes": 1,
    "duration": "0:10:00",
    "executable": "python reconstruct.py",
}
# Make experiment aware of ML model and simulation model
experiment.create_node("reconstruct-node", run_settings=reconstruct_settings)
experiment.create_model("OM4_05",path='/lus/snx11254/ashao/scratch/MOM6-examples/ice_ocean_SIS2/OM4_05/',run_settings=OM4_05_settings)
# Orchestrate the connection between the ML model and simulation
orc = experiment.create_orchestrator(cluster_size=3)
experiment.register_connection("OM4_05", "reconstruct-node")

# Launch everything and poll Slurm for statuses
experiment.start(launcher="slurm")
experiment.poll()
experiment.release()

