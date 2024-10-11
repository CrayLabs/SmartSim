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

import time
import typing as t
from collections import OrderedDict

import numpy as np

from ...log import get_logger

logger = get_logger("PerfTimer")


class PerfTimer:
    def __init__(
        self,
        filename: str = "timings",
        prefix: str = "",
        timing_on: bool = True,
        debug: bool = False,
    ):
        self._start: t.Optional[float] = None
        self._interm: t.Optional[float] = None
        self._timings: OrderedDict[str, list[t.Union[float, int, str]]] = OrderedDict()
        self._timing_on = timing_on
        self._filename = filename
        self._prefix = prefix
        self._debug = debug

    def _add_label_to_timings(self, label: str) -> None:
        if label not in self._timings:
            self._timings[label] = []

    @staticmethod
    def _format_number(number: t.Union[float, int]) -> str:
        """Formats the input value with a fixed precision appropriate for logging"""
        return f"{number:0.4e}"

    def start_timings(
        self,
        first_label: t.Optional[str] = None,
        first_value: t.Optional[t.Union[float, int]] = None,
    ) -> None:
        """Start a recording session by recording

        :param first_label: a label for an event that will be manually prepended
        to the timing information before starting timers
        :param first_label: a value for an event that will be manually prepended
        to the timing information before starting timers"""
        if self._timing_on:
            if first_label is not None and first_value is not None:
                mod_label = self._make_label(first_label)
                value = self._format_number(first_value)
                self._log(f"Started timing: {mod_label}: {value}")
                self._add_label_to_timings(mod_label)
                self._timings[mod_label].append(value)
            self._start = time.perf_counter()
            self._interm = time.perf_counter()

    def end_timings(self) -> None:
        """Record a timing event and clear the last checkpoint"""
        if self._timing_on and self._start is not None:
            mod_label = self._make_label("total_time")
            self._add_label_to_timings(mod_label)
            delta = self._format_number(time.perf_counter() - self._start)
            self._timings[self._make_label("total_time")].append(delta)
            self._log(f"Finished timing: {mod_label}: {delta}")
            self._interm = None

    def _make_label(self, label: str) -> str:
        """Return a label formatted with the current label prefix

        :param label: the original label
        :returns: the adjusted label value"""
        return self._prefix + label

    def _get_delta(self) -> float:
        """Calculates the offset from the last intermediate checkpoint time

        :returns: the number of seconds elapsed"""
        if self._interm is None:
            return 0
        return time.perf_counter() - self._interm

    def get_last(self, label: str) -> str:
        """Return the last timing value collected for the given label in
        the format `{label}: {value}`. If no timing value has been collected
        with the label, returns `Not measured yet`"""
        mod_label = self._make_label(label)
        if mod_label in self._timings:
            value = self._timings[mod_label][-1]
            if value:
                return f"{label}: {value}"

        return "Not measured yet"

    def measure_time(self, label: str) -> None:
        """Record a new time event if timing is enabled

        :param label: the label to record a timing event for"""
        if self._timing_on and self._interm is not None:
            mod_label = self._make_label(label)
            self._add_label_to_timings(mod_label)
            delta = self._format_number(self._get_delta())
            self._timings[mod_label].append(delta)
            self._log(f"{mod_label}: {delta}")
            self._interm = time.perf_counter()

    def _log(self, msg: str) -> None:
        """Conditionally logs a message when the debug flag is enabled

        :param msg: the message to be logged"""
        if self._debug:
            logger.info(msg)

    @property
    def max_length(self) -> int:
        """Returns the number of records contained in the largest timing set"""
        if len(self._timings) == 0:
            return 0
        return max(len(value) for value in self._timings.values())

    def print_timings(self, to_file: bool = False, to_stdout: bool = True) -> None:
        """Print timing information to standard output. If `to_file`
        is `True`, also write results to a file.

        :param to_file: flag indicating if timing should be written to the timing file
        :param to_stdout: flag indicating if timing should be written to stdout"""
        if to_stdout:
            print(" ".join(self._timings.keys()))
        try:
            value_array = np.array(list(self._timings.values()), dtype=float)
        except Exception as e:
            logger.exception(e)
            return
        value_array = np.transpose(value_array)
        if self._debug and to_stdout:
            for i in range(value_array.shape[0]):
                print(" ".join(self._format_number(value) for value in value_array[i]))
        if to_file:
            np.save(self._prefix + self._filename + ".npy", value_array)

    @property
    def is_active(self) -> bool:
        """Return `True` if timer is recording, `False` otherwise"""
        return self._timing_on

    @is_active.setter
    def is_active(self, active: bool) -> None:
        """Set to `True` to record timing information, `False` otherwise"""
        self._timing_on = active
