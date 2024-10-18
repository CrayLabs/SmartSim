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

import os
from shutil import which

import pytest

import smartsim.wlm as wlm
from smartsim.error.errors import LauncherError, SmartSimError, SSUnsupportedError

# alloc_specs can be specified by the user when testing, but it will
# require all WLM env variables to be populated. If alloc_specs is not
# defined, the tests in this file are skipped.


def test_get_hosts(alloc_specs):
    if not alloc_specs:
        pytest.skip("alloc_specs not defined")

    def verify_output(output):
        assert isinstance(output, list)
        assert all(isinstance(host, str) for host in output)
        if "host_list" in alloc_specs:
            assert output == alloc_specs["host_list"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_hosts())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_hosts()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ:
            verify_output(wlm.get_hosts())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_hosts()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_hosts(launcher=pytest.test_launcher)


def test_get_queue(alloc_specs):
    if not alloc_specs:
        pytest.skip("alloc_specs not defined")

    def verify_output(output):
        assert isinstance(output, str)
        if "queue" in alloc_specs:
            assert output == alloc_specs["queue"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_queue())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_queue()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ:
            verify_output(wlm.get_queue())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_queue()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_queue(launcher=pytest.test_launcher)


def test_get_tasks(alloc_specs):
    if not alloc_specs:
        pytest.skip("alloc_specs not defined")

    def verify_output(output):
        assert isinstance(output, int)
        if "num_tasks" in alloc_specs:
            assert output == alloc_specs["num_tasks"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_tasks())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks(launcher=pytest.test_launcher)

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ and which("qstat"):
            verify_output(wlm.get_tasks())
        elif "PBS_JOBID" in os.environ:
            with pytest.raises(LauncherError):
                wlm.get_tasks()
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_tasks(launcher=pytest.test_launcher)


def test_get_tasks_per_node(alloc_specs):
    if not alloc_specs:
        pytest.skip("alloc_specs not defined")

    def verify_output(output):
        assert isinstance(output, dict)
        assert all(
            isinstance(node, str) and isinstance(ntasks, int)
            for node, ntasks in output.items()
        )
        if "tasks_per_node" in alloc_specs:
            assert output == alloc_specs["tasks_per_node"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_tasks_per_node())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks_per_node()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ and which("qstat"):
            verify_output(wlm.get_tasks_per_node())
        elif "PBS_JOBID" in os.environ:
            with pytest.raises(LauncherError):
                wlm.get_tasks_per_node()
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks_per_node()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_tasks_per_node(launcher=pytest.test_launcher)
