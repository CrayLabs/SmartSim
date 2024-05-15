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

import typing as t

if t.TYPE_CHECKING:
    # pylint: disable-next=unused-import
    import smartsim.settings.base


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
        :param run_settings: Launcher settings specified in the experiment
                             entity
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
