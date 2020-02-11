from smartsim import Controller, Generator, State

# initialize State
state = State(experiment="double_gyre")

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
run_params = {"nodes":2,
              "executable":"MOM6",
              "partition": "iv24"}
state.create_ensemble("quar-deg", params=quar_deg_params, run_settings=run_params)
state.create_ensemble("half-deg", params=half_deg_params, run_settings=run_params)

# Generate Models
gen = Generator(state, model_files="./MOM6_base_config")
gen.generate()

sim = Controller(state, launcher="slurm")
sim.start()
sim.poll()
sim.release()
