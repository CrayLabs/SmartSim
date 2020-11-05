import sys
import logging
import coloredlogs
from ..error import SSConfigError
from ..utils.helpers import get_env


def _get_log_level():
    """Get the logging level based on environment variable
       SMARTSIM_LOG_LEVEL

       Logging levels
         - quiet: Just shows errors and warnings
         - info: Show basic information and errors (default)
         - debug: Shows info, errors and user debug information
         - developer: Shows everything happening during execution
                      extremely verbose logging.

    :return: Log level for coloredlogs
    :rtype: str
    """
    try:
        log_level = str(get_env("SMARTSIM_LOG_LEVEL"))
        if log_level == "quiet":
            return "warning"
        elif log_level == "info":
            return "info"
        elif log_level == "debug":
            return "debug"
        # extremely verbose logging used internally
        elif log_level == "developer":
            return "debug"
        else:
            return "info"
    except SSConfigError:
        return "info"


def get_logger(name=None, log_level=None):
    """Returns a log handle that has had the appropriate style
    set to ensure logging practices are consistent and clean across the
    code base.

    :param str name: the name of the desired logger.

    :param int log_level: what level to set the logger at.  Valid values are
                          defined in the python logging module.
    """
    # if name is None, then logger is the root logger
    # if not root logger, get the name of file without prefix.
    if name:
        try:
            user_log_level = str(get_env("SMARTSIM_LOG_LEVEL"))
            if user_log_level != "developer" and user_log_level != "debug":
                name = "SmartSim"
        except SSConfigError:
            name = "SmartSim"
    logger = logging.getLogger(name)
    if log_level:
        logger.setLevel(log_level)
    else:
        log_level = _get_log_level()
    coloredlogs.install(level=log_level, logger=logger, stream=sys.stdout)
    return logger

def log_to_file(filename, log_level=None):
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param str filename: the name of the desired log file.

    :param int log_level: as defiend in get_logger.  Can be specified
                          to allow the file to store more or less verbose
                          logging information.
    """
    coloredlogs.install(stream=open(filename, "w+"), log_level=log_level)