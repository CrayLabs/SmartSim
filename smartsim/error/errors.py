# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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


# Exceptions


class SmartSimError(Exception):
    """Base SmartSim error"""


class SSUnsupportedError(Exception):
    """Raised in the event that a called method isn't supported by SmartSim yet"""


class EntityExistsError(SmartSimError):
    """Raised when a user tries to create an entity or files/directories for
    an entity and either the entity/files/directories already exist"""


class UserStrategyError(SmartSimError):
    """Raised when there is an error with model creation inside an ensemble
    that is from a user provided permutation strategy"""

    def __init__(self, perm_strat):
        message = self.create_message(perm_strat)
        super().__init__(message)

    def create_message(self, perm_strat):
        prefix = "User provided ensemble generation strategy"
        message = "failed to generate valid parameter names and values"
        return " ".join((prefix, str(perm_strat), message))


class ParameterWriterError(SmartSimError):
    """Raised in the event that input parameter files for a model
    could not be written.
    """

    def __init__(self, file_path, read=True):
        message = self.create_message(file_path, read)
        super().__init__(message)

    def create_message(self, fp, read):
        if read:
            msg = f"Failed to read configuration file to write at {fp}"
        else:
            msg = f"Failed to write configuration file to {fp}"
        return msg


# Internal Exceptions


class SSInternalError(Exception):
    """
    SSInternalError is raised when an internal error is encountered.
    """

    pass


class SSConfigError(SSInternalError):
    """Raised when there is an error in the configuration of SmartSim"""


class LauncherError(SSInternalError):
    """Raised when there is an error in the launcher"""


class AllocationError(LauncherError):
    """Raised when there is a problem with the user WLM allocation"""


class ShellError(LauncherError):
    """Raised when error arises from function within launcher.shell
    Closely related to error from subprocess(Popen) commands"""

    def __init__(self, message, shell_error, command_list):
        msg = self.create_message(message, shell_error, command_list)
        super().__init__(msg)

    def create_message(self, message, shell_error, command_list):
        if isinstance(command_list, list):
            command_list = " ".join(command_list)
        msg = message + "\n"
        msg += f"\nCommand: {command_list}"
        if shell_error:
            msg += f"\nError from shell: {shell_error}"
        return msg
