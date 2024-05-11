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

import psutil
import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_db_fixtures(local_experiment, local_db, prepare_db):
    db = prepare_db(local_db).orchestrator
    local_experiment.reconnect_orchestrator(db.checkpoint_file)
    assert db.is_active()
    local_experiment.stop(db)


def test_create_new_db_fixture_if_stopped(local_experiment, local_db, prepare_db):
    # Run this twice to make sure that there is a stopped database
    output = prepare_db(local_db)
    local_experiment.reconnect_orchestrator(output.orchestrator.checkpoint_file)
    local_experiment.stop(output.orchestrator)

    output = prepare_db(local_db)
    assert output.new_db
    local_experiment.reconnect_orchestrator(output.orchestrator.checkpoint_file)
    assert output.orchestrator.is_active()
