from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
model_settings = exp.create_run_settings(exe="path/to/executable/simulation")

# Initialize a Model object
model_instance = exp.create_model("model_name", model_settings, params={"THERMO":1})

# Attach the file to the Model instance
model_instance.attach_generator_files(to_configure="path/to/params_inputs.txt")

# Store model_instance outputs within the Experiment directory named getting-started
exp.generate(model_instance)

# Launch the Model
exp.start(model_instance)