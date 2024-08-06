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

import pytest

dragon = pytest.importorskip("dragon")

import dragon.utils as du
from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import DragonFLIError, FLInterface

from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragonfli import DragonFLIChannel
from smartsim._core.mli.infrastructure.environmentloader import EnvironmentConfigLoader
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import (
    DragonFeatureStore,
)

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"a"),
        pytest.param(b"new byte string"),
    ],
)
def test_environment_loader_attach_fli(content: bytes, monkeypatch: pytest.MonkeyPatch):
    """A descriptor can be stored, loaded, and reattached"""
    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SS_QUEUE", du.B64.bytes_to_str(queue.serialize()))

    config = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )
    config_queue = config.get_queue()

    _ = config_queue.send(content)

    old_recv = queue.recvh()
    result, _ = old_recv.recv_bytes()
    assert result == content


def test_environment_loader_serialize_fli(monkeypatch: pytest.MonkeyPatch):
    """The serialized descriptors of a loaded and unloaded
    queue are the same"""
    chan = Channel.make_process_local()
    queue = FLInterface(main_ch=chan)
    monkeypatch.setenv("SS_QUEUE", du.B64.bytes_to_str(queue.serialize()))

    config = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel.from_descriptor,
        queue_factory=DragonFLIChannel.from_descriptor,
    )
    config_queue = config.get_queue()
    assert config_queue._fli.serialize() == queue.serialize()


def test_environment_loader_flifails(monkeypatch: pytest.MonkeyPatch):
    """An incorrect serialized descriptor will fails to attach"""
    monkeypatch.setenv("SS_QUEUE", "randomstring")
    config = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=None,
        queue_factory=DragonFLIChannel.from_descriptor,
    )

    with pytest.raises(DragonFLIError):
        config.get_queue()


def test_environment_loader_backbone_load_dfs(monkeypatch: pytest.MonkeyPatch):
    """Verify the dragon feature store is loaded correctly by the
    EnvironmentConfigLoader to demonstrate featurestore_factory correctness"""
    feature_store = DragonFeatureStore(DDict())
    monkeypatch.setenv("SS_INFRA_BACKBONE", feature_store.descriptor)

    config = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=None,
        queue_factory=None,
    )

    print(f"calling config.get_backbone: `{feature_store.descriptor}`")

    backbone = config.get_backbone()
    assert backbone is not None


def test_environment_variables_not_set():
    """EnvironmentConfigLoader getters return None when environment
    variables are not set"""
    config = EnvironmentConfigLoader(
        featurestore_factory=DragonFeatureStore.from_descriptor,
        callback_factory=DragonCommChannel.from_descriptor,
        queue_factory=DragonCommChannel.from_descriptor,
    )
    assert config.get_backbone() is None
    assert config.get_queue() is None
