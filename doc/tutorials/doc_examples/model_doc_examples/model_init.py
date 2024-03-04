from smartsim import Experiment

# Init Experiment and specify to launch locally in this example
exp = Experiment(name="getting-started", launcher="local")

# Initialize RunSettings
model_settings = exp.create_run_settings(exe="echo", exe_args="Hello World")

# Initialize Model instance
model_instance = exp.create_model(name="example-model", run_settings=model_settings)

# Generate Model directory
exp.generate(model_instance)

# Launch Model
exp.start(model_instance)