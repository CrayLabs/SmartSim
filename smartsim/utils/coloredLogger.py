import logging
import coloredlogs

def _get_logger(name=__name__, log_level="DEV"):
    logger = logging.getLogger(name)
    if log_level == "DEV":
        coloredlogs.install(level=log_level)
    else:
        coloredlogs.install(level=log_level, logger=logger)
    return logger