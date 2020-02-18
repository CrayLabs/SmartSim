
"""
A file of helper functions for SmartSim
"""
from ..error import SSConfigError
from os import environ

def get_env(env_var):
    """Retrieve an environment variable through os.environ

       :param str env_var: environment variable to retrieve
       :throws: SSConfigError
    """
    try:
        value = environ[env_var]
        return value
    except KeyError:
        raise SSConfigError("SmartSim environment not set up!")