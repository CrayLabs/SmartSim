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

import time
import typing as t

Seconds = t.NewType("Seconds", float)


class SynchronousTimeInterval:
    """A utility class to represent and synchronously block the execution of a
    thread for an interval of time.
    """

    def __init__(self, delta: float | None) -> None:
        """Initialize a new `SynchronousTimeInterval` interval

        :param delta: The difference in time the interval represents in
            seconds. If `None`, the interval will represent an infinite amount
            of time.
        :raises ValueError: The `delta` is negative
        """
        if delta is not None and delta < 0:
            raise ValueError("Timeout value cannot be less than 0")
        if delta is None:
            delta = float("inf")
        self._delta = Seconds(delta)
        """The amount of time, in seconds, the interval spans."""
        self._start = time.perf_counter()
        """The time of the creation of the interval"""

    @property
    def delta(self) -> Seconds:
        """The difference in time the interval represents

        :returns: The difference in time the interval represents
        """
        return self._delta

    @property
    def elapsed(self) -> Seconds:
        """The amount of time that has passed since the interval was created

        :returns: The amount of time that has passed since the interval was
            created
        """
        return Seconds(time.perf_counter() - self._start)

    @property
    def remaining(self) -> Seconds:
        """The amount of time remaining in the interval

        :returns: The amount of time remaining in the interval
        """
        return Seconds(max(self.delta - self.elapsed, 0))

    @property
    def expired(self) -> bool:
        """The amount of time remaining in interval

        :returns: The amount of time left in the interval
        """
        return self.remaining <= 0

    @property
    def infinite(self) -> bool:
        """Return true if the timeout interval is infinitely long

        :returns: `True` if the delta is infinite, `False` otherwise
        """
        return self.remaining == float("inf")

    def new_interval(self) -> SynchronousTimeInterval:
        """Make a new timeout with the same interval

        :returns: The new time interval
        """
        return type(self)(self.delta)

    def block(self) -> None:
        """Block the thread until the timeout completes

        :raises RuntimeError: The thread would be blocked forever
        """
        if self.remaining == float("inf"):
            raise RuntimeError("Cannot block thread forever")
        time.sleep(self.remaining)
