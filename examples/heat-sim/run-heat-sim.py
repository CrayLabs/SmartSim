
from smartsim import State, Controller, Generator


state = State(experiment="heat-sim")

run_settings = {
    "nodes": 150,
    "ppn": 24,
    "exe_args": "heat_file.xml heat.out 60 60 2000 2000 20 10",
    "executable": "/lus/snx11254/mellis/Heat-Simulation/heat_transfer",
    "partition": "iv24"
}

state.create_model("heat_transfer", run_settings=run_settings)
state.create_orchestrator(cluster_size=16, partition="knl")

generator = Generator(state,
                      model_files=["./heat_file.xml"])
generator.generate()

control = Controller(state, launcher="slurm")
control.start()
control.poll()