from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a BatchSettings object
bs = exp.create_batch_settings(nodes=4,
                               time="10:00:00")

# Initialize and configure a RunSettings object
rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")
rs.set_nodes(4)

# Initialize an Ensemble
ensemble = exp.create_ensemble("ensemble-replica", replicas=4, run_settings=rs, batch_settings=bs)