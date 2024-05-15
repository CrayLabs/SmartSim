from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Create the run settings for the Model
model_settings = exp.create_run_settings(exe="path/to/executable/simulation")

# Create a Model instance named 'model'
model = exp.create_model("model_name", model_settings)
# Enable tensor, Dataset and list prefixing on the 'model' instance
model.enable_key_prefixing()