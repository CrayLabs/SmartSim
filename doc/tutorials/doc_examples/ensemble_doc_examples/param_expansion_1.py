from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a RunSettings
rs = exp.create_run_settings(exe="path/to/example_simulation_program")

#Create the parameters to expand to the Ensemble members
params = {
            "name": ["Ellie", "John"],
            "parameter": [2, 11]
        }

# Initialize the Ensemble by specifying RunSettings, the params and "all_perm"
ensemble = exp.create_ensemble("model_member", run_settings=rs, params=params, perm_strategy="all_perm")
