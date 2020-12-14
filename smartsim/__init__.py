
# -*- coding: utf-8 -*-
__version__ = "0.1.0"
import os
import sys

if sys.version_info < (3,7):
    sys.exit('Python 3.7 or greater must be used with SmartSim.')

## setup smartsim environment
__library_path = os.path.abspath(__file__) + "/../../"
os.environ["SMARTSIMHOME"] = os.path.realpath(__library_path)

# Main API module
from .experiment import Experiment

# Slurm helpers
from .launcher import slurm
