from smartsim import Experiment

def timestwo(x):
    return 2*x

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
ensemble_settings = exp.create_run_settings(exe="path/to/example_simulation_program")

# Initialize a Ensemble object
ensemble_instance = exp.create_ensemble("ensemble_name", ensemble_settings)

# Attach TorchScript to Ensemble
ensemble_instance.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)