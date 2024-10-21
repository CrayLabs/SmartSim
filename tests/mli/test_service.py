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

import datetime
import multiprocessing as mp
import pathlib
import time
import typing as t
from asyncore import loop

import pytest
import torch

import smartsim.error as sse
from smartsim._core.entrypoints.service import Service

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_a


class SimpleService(Service):
    """Mock implementation of a service that counts method invocations
    using the base class event hooks."""

    def __init__(
        self,
        log: t.List[str],
        quit_after: int = -1,
        as_service: bool = False,
        cooldown: float = 0,
        loop_delay: float = 0,
        hc_freq: float = -1,
        run_for: float = 0,
    ) -> None:
        super().__init__(as_service, cooldown, loop_delay, hc_freq)
        self._log = log
        self._quit_after = quit_after
        self.num_starts = 0
        self.num_shutdowns = 0
        self.num_health_checks = 0
        self.num_cooldowns = 0
        self.num_delays = 0
        self.num_iterations = 0
        self.num_can_shutdown = 0
        self.run_for = run_for
        self.start_time = time.time()

    @property
    def runtime(self) -> float:
        return time.time() - self.start_time

    def _can_shutdown(self) -> bool:
        self.num_can_shutdown += 1

        if self._quit_after > -1 and self.num_iterations >= self._quit_after:
            return True
        if self.run_for > 0:
            return self.runtime >= self.run_for

    def _on_start(self) -> None:
        self.num_starts += 1

    def _on_shutdown(self) -> None:
        self.num_shutdowns += 1

    def _on_health_check(self) -> None:
        self.num_health_checks += 1

    def _on_cooldown_elapsed(self) -> None:
        self.num_cooldowns += 1

    def _on_delay(self) -> None:
        self.num_delays += 1

    def _on_iteration(self) -> None:
        self.num_iterations += 1

        return self.num_iterations >= self._quit_after


def test_service_init() -> None:
    """Verify expected default values after Service initialization"""
    activity_log: t.List[str] = []
    service = SimpleService(activity_log)

    assert service._as_service is False
    assert service._cooldown == 0
    assert service._loop_delay == 0


def test_service_run_once() -> None:
    """Verify the service completes after a single call to _on_iteration"""
    activity_log: t.List[str] = []
    service = SimpleService(activity_log)

    service.execute()

    assert service.num_iterations == 1
    assert service.num_starts == 1
    assert service.num_cooldowns == 0  # it never exceeds a cooldown period
    assert service.num_can_shutdown == 0  # it automatically exits in run once
    assert service.num_shutdowns == 1


@pytest.mark.parametrize(
    "num_iterations",
    [
        pytest.param(0, id="Immediate Shutdown"),
        pytest.param(1, id="1x"),
        pytest.param(2, id="2x"),
        pytest.param(4, id="4x"),
        pytest.param(8, id="8x"),
        pytest.param(16, id="16x"),
        pytest.param(32, id="32x"),
    ],
)
def test_service_run_until_can_shutdown(num_iterations: int) -> None:
    """Verify the service completes after a dynamic number of iterations
    based on the return value of `_can_shutdown`"""
    activity_log: t.List[str] = []

    service = SimpleService(activity_log, quit_after=num_iterations, as_service=True)

    service.execute()

    if num_iterations == 0:
        # no matter what, it should always execute the _on_iteration method
        assert service.num_iterations == 1
    else:
        # the shutdown check follows on_iteration. there will be one last call
        assert service.num_iterations == num_iterations

    assert service.num_starts == 1
    assert service.num_shutdowns == 1


@pytest.mark.parametrize(
    "cooldown",
    [
        pytest.param(1, id="1s"),
        pytest.param(3, id="3s"),
        pytest.param(5, id="5s"),
    ],
)
def test_service_cooldown(cooldown: int) -> None:
    """Verify that the cooldown period is respected"""
    activity_log: t.List[str] = []

    service = SimpleService(
        activity_log,
        quit_after=1,
        as_service=True,
        cooldown=cooldown,
        loop_delay=0,
    )

    ts0 = datetime.datetime.now()
    service.execute()
    ts1 = datetime.datetime.now()

    fudge_factor = 1.1  # allow a little bit of wiggle room for the loop
    duration_in_seconds = (ts1 - ts0).total_seconds()

    assert duration_in_seconds <= cooldown * fudge_factor
    assert service.num_cooldowns == 1
    assert service.num_shutdowns == 1


@pytest.mark.parametrize(
    "delay, num_iterations",
    [
        pytest.param(1, 3, id="1s delay, 3x"),
        pytest.param(3, 2, id="2s delay, 2x"),
        pytest.param(5, 1, id="5s delay, 1x"),
    ],
)
def test_service_delay(delay: int, num_iterations: int) -> None:
    """Verify that a delay is correctly added between iterations"""
    activity_log: t.List[str] = []

    service = SimpleService(
        activity_log,
        quit_after=num_iterations,
        as_service=True,
        cooldown=0,
        loop_delay=delay,
    )

    ts0 = datetime.datetime.now()
    service.execute()
    ts1 = datetime.datetime.now()

    # the expected duration is the sum of the delay between each iteration
    expected_duration = (num_iterations + 1) * delay
    duration_in_seconds = (ts1 - ts0).total_seconds()

    assert duration_in_seconds <= expected_duration
    assert service.num_cooldowns == 0
    assert service.num_shutdowns == 1


@pytest.mark.parametrize(
    "health_check_freq, run_for",
    [
        pytest.param(1, 5.5, id="1s freq, 10x"),
        pytest.param(5, 10.5, id="5s freq, 2x"),
        pytest.param(0.1, 5.1, id="0.1s freq, 50x"),
    ],
)
def test_service_health_check_freq(health_check_freq: float, run_for: float) -> None:
    """Verify that a the health check frequency is honored

    :param health_check_freq: The desired frequency of the health check
    :pram run_for: A fixed duration to allow the service to run
    """
    activity_log: t.List[str] = []

    service = SimpleService(
        activity_log,
        quit_after=-1,
        as_service=True,
        cooldown=0,
        hc_freq=health_check_freq,
        run_for=run_for,
    )

    ts0 = datetime.datetime.now()
    service.execute()
    ts1 = datetime.datetime.now()

    # the expected duration is the sum of the delay between each iteration
    expected_hc_count = run_for // health_check_freq

    # allow some wiggle room for frequency comparison
    assert expected_hc_count - 2 <= service.num_health_checks <= expected_hc_count + 2

    assert service.num_cooldowns == 0
    assert service.num_shutdowns == 1


def test_service_health_check_freq_unbound() -> None:
    """Verify that a health check frequency of zero is treated as
    "always on" and is called each loop iteration

    :param health_check_freq: The desired frequency of the health check
    :pram run_for: A fixed duration to allow the service to run
    """
    health_check_freq: float = 0.0
    run_for: float = 5

    activity_log: t.List[str] = []

    service = SimpleService(
        activity_log,
        quit_after=-1,
        as_service=True,
        cooldown=0,
        hc_freq=health_check_freq,
        run_for=run_for,
    )

    service.execute()

    # allow some wiggle room for frequency comparison
    assert service.num_health_checks == service.num_iterations
    assert service.num_cooldowns == 0
    assert service.num_shutdowns == 1
