
# -*- coding: utf-8 -*-
__version__ = "0.1.0"

import sys
if sys.version_info < (3,7):
    sys.exit('Python 3.7 or greater must be used with SmartSim.')

# Main modules
from .experiment import Experiment
from .clients import Client
from .mpo import MPO