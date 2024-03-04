from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe="path/to/executable/simulation")

# Initialize a Model object
model_instance = exp.create_model("model_name", model_settings)

# TorchScript string
torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

# Attach TorchScript to Model
model_instance.add_script(name="example_script", script=torch_script_str, device="GPU", devices_per_node=2, first_device=0)