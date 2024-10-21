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

import os
import socket
import typing as t

import pytest

dragon = pytest.importorskip("dragon")

# isort: off

import dragon.infrastructure.policy as dragon_policy
import dragon.infrastructure.process_desc as dragon_process_desc
import dragon.native.process as dragon_process

# isort: on

from smartsim.log import get_logger

logger = get_logger(__name__)


def function_as_dragon_proc(
    entrypoint_fn: t.Callable[[t.Any], None],
    args: t.List[t.Any],
    cpu_affinity: t.List[int],
    gpu_affinity: t.List[int],
) -> dragon_process.Process:
    """Execute a function as an independent dragon process.

    :param entrypoint_fn: The function to execute
    :param args: The arguments for the entrypoint function
    :param cpu_affinity: The cpu affinity for the process
    :param gpu_affinity: The gpu affinity for the process
    :returns: The dragon process handle
    """
    options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
    local_policy = dragon_policy.Policy(
        placement=dragon_policy.Policy.Placement.HOST_NAME,
        host_name=socket.gethostname(),
        cpu_affinity=cpu_affinity,
        gpu_affinity=gpu_affinity,
    )
    return dragon_process.Process(
        target=entrypoint_fn,
        args=args,
        cwd=os.getcwd(),
        policy=local_policy,
        options=options,
        stderr=dragon_process.Popen.STDOUT,
        stdout=dragon_process.Popen.STDOUT,
    )
