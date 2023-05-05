# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import pytest

from smartsim.error import AllocationError
from smartsim.wlm import slurm

# retrieved from pytest fixtures
if pytest.test_launcher != "slurm":
    pytestmark = pytest.mark.skip(reason="Test is only for Slurm WLM systems")


def test_get_release_allocation(wlmutils):
    """test slurm interface for obtaining allocations"""
    account = wlmutils.get_test_account()
    alloc = slurm.get_allocation(nodes=1, time="00:05:00", account=account)
    time.sleep(5)  # give slurm a rest
    slurm.release_allocation(alloc)


def test_get_release_allocation_w_options(wlmutils):
    """test slurm interface for obtaining allocations"""
    options = {"ntasks-per-node": 1}
    account = wlmutils.get_test_account()
    alloc = slurm.get_allocation(nodes=1, time="00:05:00", options=options, account=account)
    time.sleep(5)  # give slurm a rest
    slurm.release_allocation(alloc)


# --------- Error handling ----------------------------


def test_release_non_existant_alloc():
    """Release allocation that doesn't exist"""
    with pytest.raises(AllocationError):
        slurm.release_allocation(00000000)
