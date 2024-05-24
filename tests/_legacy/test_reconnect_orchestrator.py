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

import os.path as osp
import time

import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.status import SmartSimStatus

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


first_dir = ""

# TODO ensure database is shutdown
# use https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test


def test_local_orchestrator(test_dir, wlmutils):
    """Test launching orchestrator locally"""
    global first_dir
    exp_name = "test-orc-launch-local"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)
    first_dir = test_dir

    orc = Orchestrator(port=wlmutils.get_test_port())
    orc.set_path(osp.join(test_dir, "orchestrator"))

    exp.start(orc)
    statuses = exp.get_status(orc)
    assert [stat != SmartSimStatus.STATUS_FAILED for stat in statuses]

    # simulate user shutting down main thread
    exp._control._jobs.actively_monitoring = False
    exp._control._launcher.task_manager.actively_monitoring = False


def test_reconnect_local_orc(test_dir):
    """Test reconnecting to orchestrator from first experiment"""
    global first_dir
    # start new experiment
    exp_name = "test-orc-local-reconnect-2nd"
    exp_2 = Experiment(exp_name, launcher="local", exp_path=test_dir)

    checkpoint = osp.join(first_dir, "orchestrator", "smartsim_db.dat")
    reloaded_orc = exp_2.reconnect_orchestrator(checkpoint)

    # let statuses update once
    time.sleep(5)

    statuses = exp_2.get_status(reloaded_orc)
    for stat in statuses:
        if stat == SmartSimStatus.STATUS_FAILED:
            exp_2.stop(reloaded_orc)
            assert False
    exp_2.stop(reloaded_orc)
