
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


def get_config(param, aux, none_ok=False):
    """Search for a configuration parameter in the initialization
        kwargs of an object Also search through an auxiliry dictionary
        in some cases.

        :param str param: parameter to search for
        :param dict aux: auxiliry dictionary to search through (default=None)
        :param bool none_ok: ok to return none if param is not present (default=False)
        :raises KeyError:
        :returns: param if present
    """
    if param in aux.keys():
        return aux[param]
    else:
        if none_ok:
            return None
        else:
            raise KeyError(param)


def remove_env(env_var):
    """Remove a variable from the environment.

    :param env_var: variable to remote
    :type env_var: string
    """

    try:
        environ.pop(env_var)
        return
    except KeyError:
        return