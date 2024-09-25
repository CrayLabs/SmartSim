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
import time
import typing as t
from abc import ABC, abstractmethod

from smartsim.log import get_logger

logger = get_logger(__name__)


class Service(ABC):
    """Base contract for standalone entrypoint scripts. Defines API for entrypoint
    behaviors (event loop, automatic shutdown, cooldown) as well as simple
    hooks for status changes"""

    def __init__(
        self, as_service: bool = False, cooldown: int = 0, loop_delay: int = 0
    ) -> None:
        """Initialize the ServiceHost
        :param as_service: Determines if the host will run until shutdown criteria
        are met or as a run-once instance
        :param cooldown: Period of time to allow service to run before automatic
        shutdown, in seconds. A non-zero, positive integer.
        :param loop_delay: delay between iterations of the event loop"""
        self._as_service = as_service
        """If the service should run until shutdown function returns True"""
        self._cooldown = abs(cooldown)
        """Duration of a cooldown period between requests to the service
        before shutdown"""
        self._loop_delay = abs(loop_delay)
        """Forced delay between iterations of the event loop"""

    @abstractmethod
    def _on_iteration(self) -> None:
        """The user-defined event handler. Executed repeatedly until shutdown
        conditions are satisfied and cooldown is elapsed.
        """

    @abstractmethod
    def _can_shutdown(self) -> bool:
        """Return true when the criteria to shut down the service are met."""

    def _on_start(self) -> None:
        """Empty hook method for use by subclasses. Called on initial entry into
        ServiceHost `execute` event loop before `_on_iteration` is invoked."""
        logger.debug(f"Starting {self.__class__.__name__}")

    def _on_shutdown(self) -> None:
        """Empty hook method for use by subclasses. Called immediately after exiting
        the main event loop during automatic shutdown."""
        logger.debug(f"Shutting down {self.__class__.__name__}")

    def _on_cooldown_elapsed(self) -> None:
        """Empty hook method for use by subclasses. Called on every event loop
        iteration immediately upon exceeding the cooldown period"""
        logger.debug(f"Cooldown exceeded by {self.__class__.__name__}")

    def _on_delay(self) -> None:
        """Empty hook method for use by subclasses. Called on every event loop
        iteration immediately before executing a delay before the next iteration"""
        logger.debug(f"Service iteration waiting for {self.__class__.__name__}s")

    def _log_cooldown(self, elapsed: float) -> None:
        """Log the remaining cooldown time, if any"""
        remaining = self._cooldown - elapsed
        if remaining > 0:
            # logger.debug(f"{abs(remaining):.2f}s remains of {self._cooldown}s cooldown")
            ...
        else:
            logger.info(f"exceeded cooldown {self._cooldown}s by {abs(remaining):.2f}s")

    def execute(self) -> None:
        """The main event loop of a service host. Evaluates shutdown criteria and
        combines with a cooldown period to allow automatic service termination.
        Responsible for executing calls to subclass implementation of `_on_iteration`"""
        self._on_start()

        running = True
        cooldown_start: t.Optional[datetime.datetime] = None

        while running:
            self._on_iteration()

            # allow immediate shutdown if not set to run as a service
            if not self._as_service:
                running = False
                continue

            # reset cooldown period if shutdown criteria are not met
            if not self._can_shutdown():
                cooldown_start = None

            # start tracking cooldown elapsed once eligible to quit
            if cooldown_start is None:
                cooldown_start = datetime.datetime.now()

            # change running state if cooldown period is exceeded
            if self._cooldown > 0:
                elapsed = datetime.datetime.now() - cooldown_start
                running = elapsed.total_seconds() < self._cooldown
                self._log_cooldown(elapsed.total_seconds())
                if not running:
                    self._on_cooldown_elapsed()
            elif self._cooldown < 1 and self._can_shutdown():
                running = False

            if self._loop_delay:
                self._on_delay()
                time.sleep(self._loop_delay)

        self._on_shutdown()
