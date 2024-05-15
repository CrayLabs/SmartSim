from smartsim import Experiment

# Initialize the Experiment and set the launcher to auto
exp = Experiment("getting-started", launcher="auto")

# Initialize a BatchSettings
bs = exp.create_batch_settings(nodes=2,
                               time="10:00:00")

# Initialize and configure RunSettings
rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")
rs.set_nodes(1)

#Create the parameters to expand to the Ensemble members
params = {
            "name": ["Ellie", "John"],
            "parameter": [2, 11]
        }

# Initialize the Ensemble by specifying RunSettings, BatchSettings, the params and "step"
ensemble = exp.create_ensemble("ensemble", run_settings=rs, batch_settings=bs, params=params, perm_strategy="step")