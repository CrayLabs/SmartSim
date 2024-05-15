from smartsim import Experiment

# Initialize the Experiment
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings object
ensemble_settings = exp.create_run_settings(exe="python", exe_args="/path/to/application.py")

# Initialize an Ensemble object via replicas strategy
example_ensemble = exp.create_ensemble("ensemble", ensemble_settings, replicas=2, params={"THERMO":1})

# Attach the file to the Ensemble instance
example_ensemble.attach_generator_files(to_configure="path/to/params_inputs.txt")

# Generate the Ensemble directory
exp.generate(example_ensemble)

# Launch the Ensemble
exp.start(example_ensemble)

