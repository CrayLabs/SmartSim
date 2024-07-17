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


class PerfTimer:
    def __init__(self, filename: str = "timings", prefix: str = ""):
        self._start: t.Optional[float] = None
        self._interm: t.Optional[float] = None
        self._timings: OrderedDict[str, list[t.Union[float, int, str]]] = OrderedDict()
        self._timing_on = True
        self._filename = filename
        self._prefix = prefix

    def _add_label_to_timings(self, label: str) -> None:
        if label not in self._timings:
            self._timings[label] = []

    @staticmethod
    def _format_number(number: float | int) -> str:
        return f"{number:0.4e}"

    def start_timings(
        self,
        first_label: t.Optional[str] = None,
        first_value: t.Optional[float | int] = None,
    ) -> None:
        if self._timing_on:
            if first_label is not None and first_value is not None:
                self._add_label_to_timings(self._make_label(first_label))
                self._timings[self._make_label(first_label)].append(first_value)
            self._start = time.perf_counter()
            self._interm = time.perf_counter()

    def end_timings(self) -> None:
        if self._timing_on and self._start is not None:
            self._add_label_to_timings(self._make_label("total_time"))
            self._timings[self._make_label("total_time")].append(
                self._format_number(time.perf_counter() - self._start)
            )
            self._interm = None

    def _make_label(self, label: str) -> str:
        return self._prefix + label

    def measure_time(self, label: str) -> None:
        if self._timing_on and self._interm is not None:
            self._add_label_to_timings(self._make_label(label))
            self._timings[self._make_label(label)].append(
                self._format_number(time.perf_counter() - self._interm)
            )
            self._interm = time.perf_counter()

    def print_timings(self, to_file: bool = False) -> None:
        print(" ".join(self._timings.keys()))
        value_array = np.array(list(self._timings.values()), dtype=float)
        value_array = np.transpose(value_array)
        for i in range(value_array.shape[0]):
            print(" ".join(self._format_number(value) for value in value_array[i]))
        if to_file:
            np.save(self._prefix + self._filename + ".npy", value_array)
