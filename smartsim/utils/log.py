# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import sys

import coloredlogs

from ..config import CONFIG
from ..error import SSConfigError

# constants for logging
coloredlogs.DEFAULT_DATE_FORMAT = "%H:%M:%S"
coloredlogs.DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"
)
# optional thread name logging for debugging
# coloredlogs.DEFAULT_LOG_FORMAT = '%(asctime)s [%(threadName)s] %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s'


def _get_log_level():
    """Get the logging level based on environment variable
       SMARTSIM_LOG_LEVEL or SmartSim config

       Logging levels
         - quiet: Just shows errors and warnings
         - info: Show basic information and errors (default)
         - debug: Shows info, errors and user debug information
         - developer: Shows everything happening during execution
                      extremely verbose logging.

    :return: Log level for coloredlogs
    :rtype: str
    """
    log_level = CONFIG.log_level
    if log_level == "quiet":
        return "warning"
    if log_level == "info":
        return "info"
    if log_level == "debug":
        return "debug"
    # extremely verbose logging used internally
    if log_level == "developer":
        return "debug"
    return "info"


def get_logger(name=None, log_level=None):
    """Returns a log handle that has had the appropriate style
    set to ensure logging practices are consistent and clean across the
    code base.

    :param name: the name of the desired logger
    :type name: str

    :param log_level: what level to set the logger to
    """
    # if name is None, then logger is the root logger
    # if not root logger, get the name of file without prefix.
    if name:
        try:
            user_log_level = CONFIG.log_level
            if user_log_level not in ("developer"):
                name = "SmartSim"
        except SSConfigError:  # pragma: no cover
            name = "SmartSim"
    logger = logging.getLogger(name)
    if log_level:
        logger.setLevel(log_level)
    else:
        log_level = _get_log_level()
    coloredlogs.install(level=log_level, logger=logger, stream=sys.stdout)
    return logger


def log_to_file(filename, log_level="debug"):
    """Installs a second filestream handler to the root logger,
    allowing subsequent logging calls to be sent to filename.

    :param filename: the name of the desired log file.
    :type filename: str

    :param log_level: as defiend in get_logger.  Can be specified
                      to allow the file to store more or less verbose
                      logging information.
    :type log_level: int | str
    """
    coloredlogs.install(stream=open(filename, "w+"), level=log_level)
