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
import itertools
import os
import typing as t

from smartsim.entity import strategies
from smartsim.entity.files import EntityFiles
from smartsim.entity.model import Application

if t.TYPE_CHECKING:
    import smartsim.settings.base


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO: Mocks to be removed later
# ------------------------------------------------------------------------------
# pylint: disable=multiple-statements


class _Mock:
    def __init__(self, *_: t.Any, **__: t.Any): ...
    def __getattr__(self, _: str) -> "_Mock":
        return _Mock()


# Remove with merge of #603
# https://github.com/CrayLabs/SmartSim/pull/603
class Job(_Mock): ...


# Remove with merge of #599
# https://github.com/CrayLabs/SmartSim/pull/599
class JobGroup(_Mock): ...


# Remove with merge of #587
# https://github.com/CrayLabs/SmartSim/pull/587
class LaunchSettings(_Mock): ...


# pylint: enable=multiple-statements
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


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


class SmartSimEntity:
    def __init__(
        self, name: str, path: str, run_settings: "smartsim.settings.base.RunSettings"
    ) -> None:
        """Initialize a SmartSim entity.

        Each entity must have a name, path, and
        run_settings. All entities within SmartSim
        share these attributes.

        :param name: Name of the entity
        :param path: path to output, error, and configuration files
        """
        self.name = name
        self.run_settings = run_settings
        self.path = path

    @property
    def type(self) -> str:
        """Return the name of the class"""
        return type(self).__name__

    def set_path(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError("path argument must be a string")
        self.path = path

    def __repr__(self) -> str:
        return self.name


class CompoundEntity(abc.ABC):
    @abc.abstractmethod
    def as_jobs(self, settings: LaunchSettings) -> t.Collection[Job]: ...
    def as_job_group(self, settings: LaunchSettings) -> JobGroup:
        return JobGroup(self.as_jobs(settings))


# TODO: If we like this design, we need to:
#         1) Move this to the `smartsim._core.entity.ensemble` module
#         2) Decide what to do witht the original `Ensemble` impl
class Ensemble(CompoundEntity):
    def __init__(
        self,
        name: str,
        exe: str | os.PathLike[str],
        exe_args: t.Sequence[str] | None = None,
        files: EntityFiles | None = None,
        parameters: t.Mapping[str, t.Sequence[str]] | None = None,
        permutation_strategy: str | strategies.TPermutationStrategy = "all_perm",
        max_permutations: int = 0,
        replicas: int = 1,
    ) -> None:
        self.name = name
        self.exe = os.fspath(exe)
        self.exe_args = list(exe_args) if exe_args else []
        self.files = copy.deepcopy(files) if files else EntityFiles()
        self.parameters = dict(parameters) if parameters else {}
        self.permutation_strategy = permutation_strategy
        self.max_permutations = max_permutations
        self.replicas = replicas

    def _create_applications(self) -> tuple[Application, ...]:
        permutation_strategy = strategies.resolve(self.permutation_strategy)
        permutations = permutation_strategy(self.parameters, self.max_permutations)
        permutations = permutations if permutations else [{}]
        permutations_ = itertools.chain.from_iterable(
            itertools.repeat(permutation, self.replicas) for permutation in permutations
        )
        return tuple(
            Application(
                name=f"{self.name}-{i}",
                exe=self.exe,
                run_settings=_Mock(),  # type: ignore[arg-type]
                # ^^^^^^^^^^^^^^^^^^^
                # FIXME: remove this constructor arg! It should not exist!!
                exe_args=self.exe_args,
                files=self.files,
                params=permutation,
            )
            for i, permutation in enumerate(permutations_)
        )

    def as_jobs(self, settings: LaunchSettings) -> tuple[Job, ...]:
        apps = self._create_applications()
        if not apps:
            raise ValueError("There are no members as part of this ensemble")
        return tuple(Job(app, settings) for app in apps)
