from smartsim import Controller, Generator, State

STATE = State(config="CP2K/simulation.toml")

# Data Generation Phase
GEN = Generator(STATE)
GEN.generate()

SIM = Controller(STATE)
SIM.start()
SIM.poll()
