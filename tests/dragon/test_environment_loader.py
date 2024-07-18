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
import typing as t

import pytest

from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStore
from smartsim.error.errors import SmartSimError

dragon = pytest.importorskip("dragon")

import dragon.utils as du
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import DragonFLIError, FLInterface

from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)

from .featurestore import FileSystemFeatureStore, MemoryFeatureStore

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

    new_sender = config_queue.send(content)

    old_recv = queue.recvh()
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
    assert config_queue._fli.serialize() == queue.serialize()


def test_environment_loader_FLI_fails(monkeypatch):
    """An incorrect serialized descriptor will fails to attach"""
    monkeypatch.setenv("SSQueue", "randomstring")
    config = EnvironmentConfigLoader()

    with pytest.raises(DragonFLIError):
        config_queue = config.get_queue()


@pytest.mark.parametrize(
    "feature_stores",
    [
        pytest.param([], id="No feature stores"),
        pytest.param([MemoryFeatureStore()], id="Single feature store"),
        pytest.param(
            [MemoryFeatureStore(), FileSystemFeatureStore()],
            id="Multiple feature stores",
        ),
    ],
)
def test_environment_loader_featurestores(
    feature_stores: t.List[FeatureStore], monkeypatch: pytest.MonkeyPatch
):
    """FeatureStore can be correctly identified, serialized and deserialized"""
    with monkeypatch.context() as m:
        for fs in feature_stores:
            value = base64.b64encode(pickle.dumps(fs)).decode("utf-8")
            key = f"SSFeatureStore.{fs.descriptor}"
            m.setenv(key, value)

        config = EnvironmentConfigLoader()
        actual_feature_stores = config.get_feature_stores()

        for fs in feature_stores:
            # Confirm that the descriptors were used as keys in the loaded feature stores
            assert fs.descriptor in actual_feature_stores

            # Confirm that the value loaded from env var is a FeatureStore
            # and it is consistent w/the key identifying it
            loaded_fs = actual_feature_stores[fs.descriptor]
            assert loaded_fs.descriptor == fs.descriptor


@pytest.mark.parametrize(
    "value_to_use,error_filter",
    [
        pytest.param("", "empty", id="Empty value"),
        pytest.param("abcd", "invalid", id="Incorrectly serialized value"),
    ],
)
def test_environment_loader_featurestores_errors(
    value_to_use: str, error_filter: str, monkeypatch: pytest.MonkeyPatch
):
    """Verify that the environment loader reports an error when a feature store
    env var is populated with something that cannot be loaded properly"""

    fs = FileSystemFeatureStore()  # just use for descriptor...
    key = f"SSFeatureStore.{fs.descriptor}"

    with monkeypatch.context() as m, pytest.raises(SmartSimError) as ex:
        m.setenv(key, value_to_use)  # <----- simulate incorrect value in env var

        config = EnvironmentConfigLoader()
        config.get_feature_stores()  # <---- kick off validation

    # confirm the specific key is reported in error message
    assert key in ex.value.args[0]
    # ensure the failure occurred during loading
    assert error_filter in ex.value.args[0].lower()


def test_environment_variables_not_set():
    """EnvironmentConfigLoader getters return None when environment
    variables are not set"""
    config = EnvironmentConfigLoader()
    assert config.get_feature_stores() == {}
    assert config.get_queue() == None
