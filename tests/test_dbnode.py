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


import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error.errors import SmartSimError


def test_parse_db_host_error():
    orc = Orchestrator()
    orc.entities[0].path = "not/a/path"
    # Fail to obtain database hostname
    with pytest.raises(SmartSimError):
        orc.entities[0].host


def test_hosts(fileutils, wlmutils):
    exp_name = "test_hosts"
    exp = Experiment(exp_name)
    test_dir = fileutils.make_test_dir()

    orc = Orchestrator(port=wlmutils.get_test_port(), interface="lo", launcher="local")
    orc.set_path(test_dir)
    exp.start(orc)

    thrown = False
    hosts = []
    try:
        hosts = orc.hosts
    except SmartSimError:
        thrown = True
    finally:
        # stop the database even if there is an error raised
        exp.stop(orc)
        orc.remove_stale_files()
        assert not thrown
        assert hosts == orc.hosts


def test_set_host():
    orc = Orchestrator()
    orc.entities[0].set_host("host")
    assert orc.entities[0]._host == "host"
