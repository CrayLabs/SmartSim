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

import base64
import os
import pickle

import pytest

from smartsim._core.mli.infrastructure.control.workermanager import (
    EnvironmentConfigLoader,
)

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


@pytest.mark.parametrize(
    "feature_store, queue",
    [
        pytest.param("feature_store_string", "queue_string", id="strings"),
        pytest.param(b"feature_store_string", b"queue_string", id="bytes"),
        pytest.param(123, 456, id="Integers"),
        pytest.param({"key": "value"}, {"key": 1}, id="Dicts"),
        pytest.param(["feature", "store", "list"], ["queue", "list"], id="Lists"),
        pytest.param(None, None, id="None"),
    ],
)
def test_environment_config_loader(feature_store, queue):
    os.environ["SSFeatureStore"] = base64.b64encode(pickle.dumps(feature_store)).decode(
        "utf-8"
    )
    os.environ["SSQueue"] = base64.b64encode(pickle.dumps(queue)).decode("utf-8")
    config = EnvironmentConfigLoader()
    config_store = config.get_feature_store()
    assert config_store == feature_store
    config_queue = config.get_queue()
    assert config_queue == queue
