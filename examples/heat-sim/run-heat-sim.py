
from smartsim import Experiment

experiment = Experiment("heat-sim")

run_settings = {
    "nodes": 50,
    "ppn": 24,
    "exe_args": "heat_file.xml heat.out 40 30 1000 1000 20 10",
    "executable": "/lus/snx11254/mellis/Heat-Simulation/heat_transfer",
    "partition": "iv24"
}

experiment.create_model("heat_transfer", run_settings=run_settings)
experiment.create_orchestrator(cluster_size=10, partition="iv24")
experiment.generate(model_files="./heat_file.xml")
experiment.start()
experiment.poll()
