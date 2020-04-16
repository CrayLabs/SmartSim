import logging
import coloredlogs
import sys

def get_logger(name=None, log_level=None):
    """Returns a log handle that has had the appropriate style
    set to ensure logging practices are consistent and clean across the
    code base.

    :param str name: the name of the desired logger.

    :param int log_level: what level to set the logger at.  Valid values are
                          defined in the python logging module.
    """

    # if name is None, then logger is the root logger
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level)
    coloredlogs.install(level=log_level, logger=logger, stream=sys.stdout)
    return logger

def set_global_logging_level(log_level):
    """Sets the root logging level, which subsequent loggers
    inherit from unless setLevel has been called on them.

    :param int log_level: as defined in `get_logger`.
    """

    get_logger().setLevel(log_level)

def set_logging_level(logger, log_level):
    """Sets the logging level for an individual logger.  Typically,
    a logger inherits the log_level from root (and is NOTSET).

    :param logger: the handle to the logger.

    :param int log_level: as defined in `get_logger`.
    """

    logger.setLevel(log_level)

def set_debug_mode():
    """Sets the root logger to debug mode, and clears the logging level set
    on other loggers such that they log at the debug level.
    """

    set_global_logging_level(logging.DEBUG)
    for name in logging.root.manager.loggerDict:
        # the root logger is not in this dictionary
        get_logger(name).setLevel(logging.NOTSET)

def log_to_file(filename, log_level=None):
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param str filename: the name of the desired log file.

    :param int log_level: as defiend in get_logger.  Can be specified
                          to allow the file to store more or less verbose
                          logging information.
    """
    coloredlogs.install(stream=open(filename, "w+"), log_level=log_level)