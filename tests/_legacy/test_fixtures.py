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
from smartsim.database import FeatureStore
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_db_fixtures(local_experiment, local_fs, prepare_fs):
    fs = prepare_fs(local_fs).featurestore
    local_experiment.reconnect_feature_store(fs.checkpoint_file)
    assert fs.is_active()
    local_experiment.stop(fs)


def test_create_new_fs_fixture_if_stopped(local_experiment, local_fs, prepare_fs):
    # Run this twice to make sure that there is a stopped database
    output = prepare_fs(local_fs)
    local_experiment.reconnect_feature_store(output.featurestore.checkpoint_file)
    local_experiment.stop(output.featurestore)

    output = prepare_fs(local_fs)
    assert output.new_fs
    local_experiment.reconnect_feature_store(output.featurestore.checkpoint_file)
    assert output.featurestore.is_active()
