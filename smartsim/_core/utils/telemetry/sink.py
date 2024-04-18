# BSD 2-Clause License
#
# Copyright (c) 2021-2024 Hewlett Packard Enterprise
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
import abc
import logging
import pathlib
import typing as t

logger = logging.getLogger("TelemetryMonitor")


class Sink(abc.ABC):
    """Base class for output sinks. Represents a durable, read-only
    storage mechanism"""

    @abc.abstractmethod
    async def save(self, *args: t.Any) -> None:
        """Save the args passed to this method to the underlying sink

        :param args: variadic list of values to save
        :type args:  Any"""


class FileSink(Sink):
    """Telemetry sink that writes to a file"""

    def __init__(self, path: str) -> None:
        """Initialize the FileSink

        :param filename: path to a file backing this `Sink`
        :type filename: str"""
        super().__init__()
        self._check_init(path)
        self._path = pathlib.Path(path)

    @staticmethod
    def _check_init(filename: str) -> None:
        """Validate initialization arguments and raise a ValueError
        if an invalid filename is passed

        :param filename: path to a file backing this `Sink`
        :type filename: str"""
        if not filename:
            raise ValueError("No filename provided to FileSink")

    @property
    def path(self) -> pathlib.Path:
        """The path to the file this FileSink writes

        :return: path to a file backing this `Sink`
        :rtype: pathlib.Path"""
        return self._path

    async def save(self, *args: t.Any) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._path, "a+", encoding="utf-8") as sink_fp:
            values = ",".join(map(str, args)) + "\n"
            sink_fp.write(values)
