
from state import State
from generation.generator import Generator
from control.controller import Controller

STATE = State("/MOM6/simulation.toml", log_level="DEBUG")

# Data Generation Phase
GEN = Generator(STATE)
GEN.generate()

SIM = Controller(STATE)
SIM.start()

