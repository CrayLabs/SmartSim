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

dragon = pytest.importorskip("dragon")

import dragon.utils as du
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import DragonFLIError, FLInterface

from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore

from ..mli.featurestore import MemoryFeatureStore

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"a"),
        pytest.param(b"new byte string"),
    ],
)
def test_environment_loader_attach_FLI(content, monkeypatch):
    """A descriptor can be stored, loaded, and reattached"""
    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SSQueue", du.B64.bytes_to_str(queue.serialize()))

    config = EnvironmentConfigLoader()
    config_queue = config.get_queue()

    new_sender = config_queue.sendh(use_main_as_stream_channel=True)
    new_sender.send_bytes(content)

    old_recv = queue.recvh(use_main_as_stream_channel=True)
    result, _ = old_recv.recv_bytes()
    assert result == content


def test_environment_loader_serialize_FLI(monkeypatch):
    """The serialized descriptors of a loaded and unloaded
    queue are the same"""
    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SSQueue", du.B64.bytes_to_str(queue.serialize()))

    config = EnvironmentConfigLoader()
    config_queue = config.get_queue()
    assert config_queue.serialize() == queue.serialize()


def test_environment_loader_FLI_fails(monkeypatch):
    """An incorrect serialized descriptor will fails to attach"""
    monkeypatch.setenv("SSQueue", "randomstring")
    config = EnvironmentConfigLoader()

    with pytest.raises(DragonFLIError):
        config_queue = config.get_queue()


@pytest.mark.parametrize(
    "expected_keys, expected_values",
    [
        pytest.param(["key1", "key2", "key3"], ["value1", "value2", "value3"]),
        pytest.param(["another key"], ["another value"]),
    ],
)
def test_environment_loader_memory_featurestore(
    expected_keys, expected_values, monkeypatch
):
    """MemoryFeatureStores can be correctly serialized and deserialized"""
    feature_store = MemoryFeatureStore()
    key_value_pairs = zip(expected_keys, expected_values)
    for k, v in key_value_pairs:
        feature_store[k] = v
    monkeypatch.setenv(
        "SSFeatureStore", base64.b64encode(pickle.dumps(feature_store)).decode("utf-8")
    )
    config = EnvironmentConfigLoader()
    config_feature_store = config.get_feature_store()

    for k, _ in key_value_pairs:
        assert config_feature_store[k] == feature_store[k]


@pytest.mark.parametrize(
    "expected_keys, expected_values",
    [
        pytest.param(["key1", "key2", "key3"], ["value1", "value2", "value3"]),
        pytest.param(["another key"], ["another value"]),
    ],
)
def test_environment_loader_dragon_featurestore(
    expected_keys, expected_values, monkeypatch
):
    """DragonFeatureStores can be correctly serialized and deserialized"""
    storage = DDict()
    feature_store = DragonFeatureStore(storage)
    key_value_pairs = zip(expected_keys, expected_values)
    for k, v in key_value_pairs:
        feature_store[k] = v
    monkeypatch.setenv(
        "SSFeatureStore", base64.b64encode(pickle.dumps(feature_store)).decode("utf-8")
    )
    config = EnvironmentConfigLoader()
    config_feature_store = config.get_feature_store()

    for k, _ in key_value_pairs:
        assert config_feature_store[k] == feature_store[k]


def test_environment_variables_not_set():
    """EnvironmentConfigLoader getters return None when environment
    variables are not set"""
    config = EnvironmentConfigLoader()
    assert config.get_feature_store() == None
    assert config.get_queue() == None
