from smartsim import Controller, Generator, State

# initialize State
state = State(experiment="double_gyre")

# Create targets
quar_deg_params = {"KH": [200, 400, 700],
                   "KHTH": [200, 400, 700],
                   "x_resolution": 40,
                   "y_resolution": 20,
                   "months": 1}
half_deg_params = {"KH": [200, 400, 700],
                   "KHTH": [200, 400, 700],
                   "x_resolution": 40,
                   "y_resolution": 20,
                   "months": 1}
state.create_target("quar-deg", params=quar_deg_params)
state.create_target("half-deg", params=half_deg_params)



# Generate Models
gen = Generator(state, model_files="MOM6/MOM6_base_config")
gen.generate()

control_dict = {"nodes":2,
                "executable":"MOM6",
                "run_command":"srun",
                "launcher": "slurm",
                "partition": "iv24"}
sim = Controller(state, **control_dict)
sim.start()
sim.poll()
