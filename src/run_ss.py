
from state import State
from data_generation.generator import Generator

STATE = State()

# Data Generation Phase
GEN = Generator(STATE)
GEN.generate()
