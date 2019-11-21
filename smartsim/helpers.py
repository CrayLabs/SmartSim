

import sys
import logging

from subprocess import PIPE, Popen, CalledProcessError
from os import getcwd, environ, path
from .error import SmartSimError



def get_SSHOME():
    """Retrieves SMARTSIMHOME env variable"""
    try:
        SS_HOME = environ["SMARTSIMHOME"]
        if not SS_HOME.endswith("/"):
            SS_HOME += "/"
        return SS_HOME
    except KeyError:
        raise SmartSimError("SmartSim library environment not setup!")

