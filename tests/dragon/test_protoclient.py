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

import pickle
import time
import typing as t

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel, create_local
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBroadcaster,
    OnWriteFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import dragon_ddict
from smartsim._core.mli.infrastructure.storage.feature_store import ReservedKeys
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

# isort: off
from dragon import fli
from dragon.channels import Channel

# from ..ex..high_throughput_inference.mock_app import ProtoClient
from smartsim.protoclient import ProtoClient


# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon
WORK_QUEUE_KEY = "_SMARTSIM_REQUEST_QUEUE"
logger = get_logger(__name__)


@pytest.fixture
def storage_for_dragon_fs() -> t.Dict[str, str]:
    # return dragon_ddict.DDict(1, 2, total_mem=2 * 1024**3)
    return dragon_ddict.DDict(1, 2, 4 * 1024**2)


@pytest.fixture
def the_backbone(storage_for_dragon_fs) -> BackboneFeatureStore:
    return BackboneFeatureStore(storage_for_dragon_fs, allow_reserved_writes=True)


@pytest.fixture
def the_worker_queue(the_backbone: BackboneFeatureStore) -> DragonFLIChannel:
    """a stand-in for the worker manager so a worker queue exists"""

    # create the FLI
    to_worker_channel = Channel.make_process_local()
    # to_worker_channel = create_local()
    fli_ = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    comm_channel = DragonFLIChannel(fli_, True)

    # store the descriptor in the backbone
    # the_backbone.worker_queue = comm_channel.descriptor
    the_backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] = comm_channel.descriptor

    try:
        comm_channel.send(b"foo")
    except Exception as ex:
        print(f"ohnooooo: {ex}")

    return comm_channel


@pytest.fixture
def storage_for_dragon_fs_with_req_queue(
    storage_for_dragon_fs: t.Dict[str, str]
) -> t.Dict[str, str]:
    # create a valid FLI so any call to attach does not fail
    channel_ = Channel.make_process_local()
    fli_ = fli.FLInterface(main_ch=channel_, manager_ch=None)
    comm_channel = DragonFLIChannel(fli_, True)

    storage_for_dragon_fs[WORK_QUEUE_KEY] = comm_channel.descriptor
    return storage_for_dragon_fs


@pytest.mark.parametrize(
    "wait_timeout, exp_wait_max",
    [
        # aggregate the 1+1+1 into 3 on remaining parameters
        pytest.param(1, 1 + 1 + 1, id="1s wait, 3 cycle steps"),
        pytest.param(2, 3 + 2, id="2s wait, 4 cycle steps"),
        pytest.param(4, 3 + 2 + 4, id="4s wait, 5 cycle steps"),
    ],
)
def test_protoclient_timeout(
    wait_timeout: float,
    exp_wait_max: float,
    the_backbone: BackboneFeatureStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that attempts to attach to the worker queue from the protoclient
    timeout in an appropriate amount of time. Note: due to the backoff, we verify
    the elapsed time is less than the 15s of a cycle of waits

    :param wait_timeout: a timeout for use when configuring a proto client
    :param exp_wait_max: a ceiling for the expected time spent waiting for
    the timeout
    :param the_backbone: a pre-initialized backbone featurestore for setting up
    the environment variable required by the client"""

    # NOTE: exp_wait_time maps to the cycled backoff of [.1, .5, 1, 2, 4, 8]
    # with leeway added (by allowing 1s each for the 0.1 and 0.5 steps)
    start_time = time.time()
    with monkeypatch.context() as ctx, pytest.raises(SmartSimError) as ex:
        ctx.setenv("_SMARTSIM_INFRA_BACKBONE", the_backbone.descriptor)

        ProtoClient(False, wait_timeout=wait_timeout)

    end_time = time.time()
    elapsed = end_time - start_time

    # todo: revisit. should this trigger any wait if the backbone is set above?
    # confirm that we met our timeout
    # assert elapsed > wait_timeout, f"below configured timeout {wait_timeout}"

    # confirm that the total wait time is aligned with the sleep cycle
    assert elapsed < exp_wait_max, f"above expected max wait {exp_wait_max}"


def test_protoclient_initialization_no_backbone():
    """Verify that attempting to start the client without required environment variables
    results in an exception. NOTE: Backbone env var is not set"""

    with pytest.raises(SmartSimError) as ex:
        ProtoClient(timing_on=False)

    # confirm the missing value error has been raised
    assert {"backbone", "configuration"}.issubset(set(ex.value.args[0].split(" ")))


def test_protoclient_initialization(
    the_backbone: BackboneFeatureStore,
    the_worker_queue: DragonFLIChannel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that attempting to start the client with required env vars results
    in a fully initialized client

    :param the_backbone: a pre-initialized backbone featurestore
    :param the_worker_queue: an FLI channel the client will retrieve
    from the backbone"""

    with monkeypatch.context() as ctx:
        ctx.setenv("_SMARTSIM_INFRA_BACKBONE", the_backbone.descriptor)
        # NOTE: backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] set in the_worker_queue fixture

        client = ProtoClient(timing_on=False)

        # confirm the backbone was attached correctly
        assert client._backbone is not None
        assert client._backbone.descriptor == the_backbone.descriptor

        # confirm the worker queue is created and attached correctly
        assert client._to_worker_fli is not None
        assert client._to_worker_fli.descriptor == the_worker_queue.descriptor

        # confirm the worker channels are created
        assert client._from_worker_ch is not None
        assert client._from_worker_ch.descriptor

        assert client._to_worker_ch is not None
        assert client._to_worker_ch.descriptor

        # confirm a publisher is created
        assert client._publisher is not None


def test_protoclient_write_model(
    the_backbone: BackboneFeatureStore,
    the_worker_queue: DragonFLIChannel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that writing a model using the client causes the model data to be
    written to a feature store and triggers a key-written event

    :param the_backbone: a pre-initialized backbone featurestore
    :param the_worker_queue: an FLI channel the client will retrieve
    from the backbone"""

    with monkeypatch.context() as ctx:
        ctx.setenv("_SMARTSIM_INFRA_BACKBONE", the_backbone.descriptor)
        # NOTE: backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] set in the_worker_queue fixture

        client = ProtoClient(timing_on=False)

        model_key = "my-model"
        model_bytes = b"12345"

        client.set_model(model_key, model_bytes)

        # confirm the client modified the underlying feature store
        assert client._backbone[model_key] == model_bytes

        publisher = t.cast(EventBroadcaster, client._publisher)

        # confirm the client raised the key-written event
        assert len(publisher._event_buffer) == 1

        event = t.cast(OnWriteFeatureStore, pickle.loads(publisher._event_buffer.pop()))
        assert event.descriptor == the_backbone.descriptor
        assert event.key == model_key
