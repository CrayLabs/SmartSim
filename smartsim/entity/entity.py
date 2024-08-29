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

import abc
import copy
import typing as t
from abc import ABC, abstractmethod

from smartsim.launchable.jobGroup import JobGroup
from .._core.utils.helpers import expand_exe_path
from .files import EntityFiles

if t.TYPE_CHECKING:
    from smartsim.launchable.job import Job
    from smartsim.settings.launchSettings import LaunchSettings
    from smartsim.types import TODO

    RunSettings = TODO


class TelemetryConfiguration:
    """A base class for configuraing telemetry production behavior on
    existing `SmartSimEntity` subclasses. Any class that will have
    optional telemetry collection must expose access to an instance
    of `TelemetryConfiguration` such as:

    ```
    @property
    def telemetry(self) -> TelemetryConfiguration:
        # Return the telemetry configuration for this entity.
        # :returns: Configuration object indicating the configuration
        # status of telemetry for this entity
        return self._telemetry_producer
    ```

    An instance will be used by to conditionally serialize
    values to the `RuntimeManifest`
    """

    def __init__(self, enabled: bool = False) -> None:
        """Initialize the telemetry producer and immediately call the `_on_enable` hook.

        :param enabled: flag indicating the initial state of telemetry
        """
        self._is_on = enabled

        if self._is_on:
            self._on_enable()
        else:
            self._on_disable()

    @property
    def is_enabled(self) -> bool:
        """Boolean flag indicating if telemetry is currently enabled

        :returns: `True` if enabled, `False` otherwise
        """
        return self._is_on

    def enable(self) -> None:
        """Enable telemetry for this producer"""
        self._is_on = True
        self._on_enable()

    def disable(self) -> None:
        """Disable telemetry for this producer"""
        self._is_on = False
        self._on_disable()

    def _on_enable(self) -> None:
        """Overridable hook called after telemetry is `enabled`. Allows subclasses
        to perform actions when attempts to change configuration are made"""

    def _on_disable(self) -> None:
        """Overridable hook called after telemetry is `disabled`. Allows subclasses
        to perform actions when attempts to change configuration are made"""


class SmartSimEntity(abc.ABC):
    def __init__(
        self,
        name: str,
        exe: str,
        exe_args: t.Union[str, t.Sequence[str], None],
        files: t.Union[EntityFiles, None],
    ) -> None:
        """Initialize a SmartSim entity.

        Each entity must have a name and path. All entities within SmartSim
        share these attributes.

        :param name: Name of the entity
        """
        self.name = name
        """The name of the application"""
        self._exe = expand_exe_path(exe)
        """The executable to run"""
        self._exe_args = self._build_exe_args(exe_args) or []
        """The executable arguments"""
        self._files = copy.deepcopy(files) if files else None

    @property
    def exe(self) -> str:
        """Return executable to run.

        :returns: application executable to run
        """
        return self._exe

    @exe.setter
    def exe(self, value: str) -> None:
        """Set executable to run.

        :param value: executable to run
        """
        self._exe = copy.deepcopy(value)

    @property
    def exe_args(self) -> t.MutableSequence[str]:
        """Return a list of attached executable arguments.

        :returns: application executable arguments
        """
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Union[str, t.Sequence[str], None]) -> None:
        """Set the executable arguments.

        :param value: executable arguments
        """
        self._exe_args = self._build_exe_args(value)

    @property
    def files(self) -> t.Optional[EntityFiles]:
        """Return files to be copied, symlinked, and/or configured prior to
        execution.

        :returns: files
        """
        return self._files

    @files.setter
    def files(self, value: t.Optional[EntityFiles]) -> None:
        """Set files to be copied, symlinked, and/or configured prior to
        execution.

        :param value: files
        """
        self._files = copy.deepcopy(value)
    
    @abc.abstractmethod
    def as_program_arguments(self) -> t.Sequence[str]: ...

    def add_exe_args(self, args: t.Union[str, t.List[str], None]) -> None:
        """Add executable arguments to executable

        :param args: executable arguments
        """
        args = self._build_exe_args(args)
        self._exe_args.extend(args)

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
                isinstance(exe_args, list)
                and all(isinstance(arg, str) for arg in exe_args)
            )
        ):
            raise TypeError("Executable arguments were not a list of str or a str.")

        if isinstance(exe_args, str):
            return exe_args.split()

        return exe_args

    @property
    def type(self) -> str:
        """Return the name of the class"""
        return type(self).__name__

    def __repr__(self) -> str:
        return self.name


class CompoundEntity(abc.ABC):
    """An interface to create different types of collections of launchables
    from a single set of launch settings.

    Objects that implement this interface describe how to turn their entities
    into a collection of jobs and this interface will handle coercion into
    other collections for jobs with slightly different launching behavior.
    """

    @abc.abstractmethod
    def as_jobs(self, settings: LaunchSettings) -> t.Collection[Job]: ...
    def as_job_group(self, settings: LaunchSettings) -> JobGroup:
        return JobGroup(list(self.as_jobs(settings)))
