from smartsim import Experiment

# initialize Experiment
experiment = Experiment("double_gyre")
alloc = experiment.get_allocation(nodes=8, ppn=24)

# Create ensembles
quar_deg_params = {"KH": [200, 400],
                   "KHTH": [200, 400],
                   "x_resolution": 40,
                   "y_resolution": 20,
                   "months": 1}
half_deg_params = {"KH": [200, 400],
                   "KHTH": [200, 400],
                   "x_resolution": 40,
                   "y_resolution": 20,
                   "months": 1}
run_params = {"nodes":1,
              "ppn": 24,
              "executable":"MOM6",
              "partition": "iv24",
              "alloc": alloc}
experiment.create_ensemble("quar-deg", params=quar_deg_params, run_settings=run_params)
experiment.create_ensemble("half-deg", params=half_deg_params, run_settings=run_params)

# Generate Models
experiment.generate(model_files=["./MOM6_base_config"])

# Run the experiment
experiment.start()
experiment.poll()
experiment.release()