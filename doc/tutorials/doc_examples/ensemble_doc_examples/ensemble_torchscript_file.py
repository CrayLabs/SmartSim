from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
ensemble_settings = exp.create_run_settings(exe="path/to/example_simulation_program")

# Initialize a Model object
ensemble_instance = exp.create_ensemble("ensemble_name", ensemble_settings)

# Attach TorchScript to Ensemble
ensemble_instance.add_script(name="example_script", script_path="path/to/torchscript.py", device="GPU", devices_per_node=2, first_device=0)