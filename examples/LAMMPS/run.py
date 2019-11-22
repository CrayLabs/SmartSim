from smartsim import Controller, Generator, State

STATE = State(config="./simulation.toml")

# Data Generation Phase
GEN = Generator(STATE)
GEN.generate()

SIM = Controller(STATE, duration="10:00:00")
SIM.start()
