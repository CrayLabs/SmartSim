from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize BatchSettings
bs = exp.create_batch_settings(nodes=10,
                               time="01:00:00")

# Initialize Ensemble
ensemble = exp.create_ensemble("ensemble-append", batch_settings=bs)

# Initialize RunSettings for Model 1
srun_settings_1 = exp.create_run_settings(exe=exe, exe_args="path/to/application_script_1.py")
# Initialize RunSettings for Model 2
srun_settings_2 = exp.create_run_settings(exe=exe, exe_args="path/to/application_script_2.py")
# Initialize Model 1 with RunSettings 1
model_1 = exp.create_model(name="model_1", run_settings=srun_settings_1)
# Initialize Model 2 with RunSettings 2
model_2 = exp.create_model(name="model_2", run_settings=srun_settings_2)

# Add Model member to Ensemble
ensemble.add_model(model_1)
# Add Model member to Ensemble
ensemble.add_model(model_2)