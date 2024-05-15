from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")

# Initialize the Ensemble by specifying the number of replicas and RunSettings
ensemble = exp.create_ensemble("ensemble-replica", replicas=4, run_settings=rs)