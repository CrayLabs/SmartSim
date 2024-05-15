from smartsim import Experiment

def timestwo(x):
    return 2*x

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe="path/to/example_simulation_program")

# Initialize a Model object
model_instance = exp.create_model("model_name", model_settings)

# Append TorchScript function to Model
model_instance.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)