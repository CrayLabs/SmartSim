# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

from __future__ import annotations

import collections
import copy
import textwrap
import typing as t
from os import path as osp

from .._core.generation.operations.operations import FileSysOperationSet
from .._core.utils.helpers import expand_exe_path
from ..log import get_logger
from .entity import SmartSimEntity

logger = get_logger(__name__)


# TODO: Remove this supression when we strip fileds/functionality
#       (run-settings/batch_settings/params_as_args/etc)!
# pylint: disable-next=too-many-public-methods


class Application(SmartSimEntity):
    """The Application class enables users to execute computational tasks in an
    Experiment workflow, such as launching compiled applications, running scripts,
    or performing general computational operations.

    Applications are designed to be added to Jobs, where LaunchSettings are also
    provided to inject launcher-specific behavior into the Job.
    """

    def __init__(
        self,
        name: str,
        exe: str,
        exe_args: t.Optional[t.Union[str, t.Sequence[str]]] = None,
        file_parameters: (
            t.Mapping[str, str] | None
        ) = None,  # TODO remove when Ensemble is addressed
    ) -> None:
        """Initialize an ``Application``

        Applications require a name and an executable. Optionally, users may provide
        executable arguments, files and file parameters. To create a simple Application
        that echos `Hello World!`, consider the example below:

        .. highlight:: python
        .. code-block:: python

            # Create an application that runs the 'echo' command
            my_app = Application(name="my_app", exe="echo", exe_args="Hello World!")

        :param name: name of the application
        :param exe: executable to run
        :param exe_args: executable arguments
        """
        super().__init__(name)
        """The name of the application"""
        self._exe = expand_exe_path(exe)
        """The executable to run"""
        self._exe_args = self._build_exe_args(exe_args) or []
        """The executable arguments"""
        self.files = FileSysOperationSet([])
        """Attach files"""
        self._file_parameters = (
            copy.deepcopy(file_parameters) if file_parameters else {}
        )
        """TODO MOCK until Ensemble is implemented"""
        """Files to be copied, symlinked, and/or configured prior to execution"""
        self._incoming_entities: t.List[SmartSimEntity] = []
        """Entities for which the prefix will have to be known by other entities"""
        self._key_prefixing_enabled = False
        """Unique prefix to avoid key collisions"""

    @property
    def exe(self) -> str:
        """Return the executable.

        :return: the executable
        """
        return self._exe

    @exe.setter
    def exe(self, value: str) -> None:
        """Set the executable.

        :param value: the executable
        :raises TypeError: exe argument is not int

        """
        if not isinstance(value, str):
            raise TypeError("exe argument was not of type str")

        if value == "":
            raise ValueError("exe cannot be an empty str")

        self._exe = value

    @property
    def exe_args(self) -> t.MutableSequence[str]:
        """Return the executable arguments.

        :return: the executable arguments
        """
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Union[str, t.Sequence[str], None]) -> None:
        """Set the executable arguments.

        :param value: the executable arguments
        """
        self._exe_args = self._build_exe_args(value)

    def add_exe_args(self, args: t.Union[str, t.List[str], None]) -> None:
        """Add executable arguments to executable

        :param args: executable arguments
        """
        args = self._build_exe_args(args)
        self._exe_args.extend(args)

    @property
    def file_parameters(self) -> t.Mapping[str, str]:
        """Return file parameters.

        :return: the file parameters
        """
        return self._file_parameters

    @file_parameters.setter
    def file_parameters(self, value: t.Mapping[str, str]) -> None:
        """Set the file parameters.

        :param value: the file parameters
        :raises TypeError: file_parameters argument is not a mapping of str and str
        """
        if not (
            isinstance(value, t.Mapping)
            and all(
                isinstance(key, str) and isinstance(val, str)
                for key, val in value.items()
            )
        ):
            raise TypeError(
                "file_parameters argument was not of type mapping of str and str"
            )
        self._file_parameters = copy.deepcopy(value)

    @property
    def incoming_entities(self) -> t.List[SmartSimEntity]:
        """Return incoming entities.

        :return: incoming entities
        """
        return self._incoming_entities

    @incoming_entities.setter
    def incoming_entities(self, value: t.List[SmartSimEntity]) -> None:
        """Set the incoming entities.

        :param value: incoming entities
        :raises TypeError: incoming_entities argument is not a list of SmartSimEntity
        """
        if not isinstance(value, list) or not all(
            isinstance(x, SmartSimEntity) for x in value
        ):
            raise TypeError(
                "incoming_entities argument was not of type list of SmartSimEntity"
            )

        self._incoming_entities = copy.copy(value)

    @property
    def key_prefixing_enabled(self) -> bool:
        """Return whether key prefixing is enabled for the application.

        :param value: key prefixing enabled
        """
        return self._key_prefixing_enabled

    @key_prefixing_enabled.setter
    def key_prefixing_enabled(self, value: bool) -> None:
        """Set whether key prefixing is enabled for the application.

        :param value: key prefixing enabled
        :raises TypeError: key prefixings enabled argument was not of type bool
        """
        if not isinstance(value, bool):
            raise TypeError("key_prefixing_enabled argument was not of type bool")

        self.key_prefixing_enabled = copy.deepcopy(value)

    def as_executable_sequence(self) -> t.Sequence[str]:
        """Converts the executable and its arguments into a sequence of program arguments.

        :return: a sequence of strings representing the executable and its arguments
        """
        return [self.exe, *self.exe_args]

    @staticmethod
    def _build_exe_args(exe_args: t.Union[str, t.Sequence[str], None]) -> t.List[str]:
        """Check and convert exe_args input to a desired collection format

        :param exe_args:
        :raises TypeError: if exe_args is not a list of str or str
        """
        if not exe_args:
            return []

        if not (
            isinstance(exe_args, str)
            or (
                isinstance(exe_args, collections.abc.Sequence)
                and all(isinstance(arg, str) for arg in exe_args)
            )
        ):
            raise TypeError("Executable arguments were not a list of str or a str.")

        if isinstance(exe_args, str):
            return exe_args.split()

        return list(exe_args)

    def __str__(self) -> str:  # pragma: no cover
        exe_args_str = "\n".join(self.exe_args)
        entities_str = "\n".join(str(entity) for entity in self.incoming_entities)
        return textwrap.dedent(f"""\
            Name: {self.name}
            Type: {self.type}
            Executable:
            {self.exe}
            Executable Arguments:
            {exe_args_str}
            Incoming Entities:
            {entities_str}
            Key Prefixing Enabled: {self.key_prefixing_enabled}
            """)
