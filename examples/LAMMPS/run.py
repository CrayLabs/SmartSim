from smartsim import Controller, Generator, State

STATE = State(config="LAMMPS/simulation.toml", log_level="DEBUG")

# Data Generation Phase
GEN = Generator(STATE)
GEN.generate(n_models=10)

SIM = Controller(STATE, duration="10:00:00")
SIM.start()
