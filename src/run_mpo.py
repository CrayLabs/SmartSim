
from state import State
from data_generation.generator import Generator

MPOSTATE = State()

# Data Generation Phase
GEN = Generator(MPOSTATE)
GEN.generate()
