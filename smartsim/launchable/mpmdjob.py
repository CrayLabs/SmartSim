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
from copy import deepcopy

from smartsim.entity.entity import SmartSimEntity
from smartsim.error.errors import SSUnsupportedError
from smartsim.launchable.basejob import BaseJob
from smartsim.launchable.mpmdpair import MPMDPair
from smartsim.settings.base import RunSettings


def _check_launcher(mpmd_pairs: t.List[MPMDPair]) -> None:
    """Enforce all pairs have the same launcher"""
    flag = 0
    ret = None
    for mpmd_pair in mpmd_pairs:
        if flag == 1:
            if ret == mpmd_pair.launch_settings.run_command:
                flag = 0
            else:
                raise SSUnsupportedError("MPMD pairs must all share the same launcher.")
        ret = mpmd_pair.launch_settings.run_command
        flag = 1


def _check_entity(mpmd_pairs: t.List[MPMDPair]) -> None:
    """Enforce all pairs have the same entity types"""
    flag = 0
    ret = None
    for mpmd_pair in mpmd_pairs:
        if flag == 1:
            if type(ret) == type(mpmd_pair.entity):
                flag = 0
            else:
                raise SSUnsupportedError(
                    "MPMD pairs must all share the same entity type."
                )
        ret = mpmd_pair.entity
        flag = 1


class MPMDJob(BaseJob):
    """An MPMDJob holds references to SmartSimEntity and
    LaunchSettings pairs.  It is responsible for turning
    The stored pairs into an MPMD command(s)
    """

    def __init__(self, mpmd_pairs: t.List[MPMDPair] = None) -> None:
        super().__init__()

        self._mpmd_pairs = deepcopy(mpmd_pairs) if mpmd_pairs else []
        _check_launcher(self._mpmd_pairs)
        _check_entity(self._mpmd_pairs)
        # self.warehouse_runner = MPMDJobWarehouseRunner

    @property
    def mpmd_pairs(self) -> t.List[MPMDPair]:
        return self._mpmd_pairs

    @mpmd_pairs.setter
    def mpmd_pair(self, value):
        self._mpmd_pair = deepcopy(value)

    def add_mpmd_pair(
        self, entity: SmartSimEntity, launch_settings: RunSettings
    ) -> None:
        """
        Add a mpmd pair to the mpmd job
        """
        self.mpmd_pairs.append(MPMDPair(entity, launch_settings))
        _check_launcher(self.mpmd_pairs)
        _check_entity(self.mpmd_pairs)

    def get_launch_steps(self) -> None:  # -> LaunchSteps:
        """Return the launch steps corresponding to the
        internal data.
        """
        pass
        # return MPMDJobWarehouseRunner.run(self)

    def __str__(self) -> str:  # pragma: no cover
        """returns A user-readable string of a MPMD Job"""
        for mpmd_pair in self.mpmd_pairs:
            string = "\n== MPMD Pair == \n{}\n{}\n"
            return string.format(mpmd_pair.entity, mpmd_pair.launch_settings)
        return string
