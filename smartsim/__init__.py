# -*- coding: utf-8 -*-
__version__ = "0.3.0"

import sys

if sys.version_info < (3, 7):  # pragma: no cover
    sys.exit("Python 3.7 or greater must be used with SmartSim.")

# Main API module
from .experiment import Experiment

# Slurm helpers
from .launcher import slurm
