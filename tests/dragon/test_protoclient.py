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
import pickle
import time
import typing as t
from unittest.mock import MagicMock

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.comm.broadcaster import EventBroadcaster
from smartsim._core.mli.infrastructure.comm.event import OnWriteFeatureStore
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

# isort: off
from dragon import fli
from dragon.data.ddict.ddict import DDict

# from ..ex..high_throughput_inference.mock_app import ProtoClient
from smartsim._core.mli.client.protoclient import ProtoClient


# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon
WORK_QUEUE_KEY = BackboneFeatureStore.MLI_WORKER_QUEUE
logger = get_logger(__name__)


@pytest.fixture(scope="module")
def the_worker_queue(the_backbone: BackboneFeatureStore) -> DragonFLIChannel:
    """Fixture that creates a dragon FLI channel as a stand-in for the
    worker queue created by the worker.

    :param the_backbone: The backbone feature store to update
    with the worker queue descriptor.
    :returns: The attached `DragonFLIChannel`
    """

    # create the FLI
    to_worker_channel = create_local()
    fli_ = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
    comm_channel = DragonFLIChannel(fli_)

    # store the descriptor in the backbone
    the_backbone.worker_queue = comm_channel.descriptor

    try:
        comm_channel.send(b"foo")
    except Exception as ex:
        logger.exception(f"Test send from worker channel failed", exc_info=True)

    return comm_channel


@pytest.mark.parametrize(
    "backbone_timeout, exp_wait_max",
    [
        # aggregate the 1+1+1 into 3 on remaining parameters
        pytest.param(0.5, 1 + 1 + 1, id="0.5s wait, 3 cycle steps"),
        pytest.param(2, 3 + 2, id="2s wait, 4 cycle steps"),
        pytest.param(4, 3 + 2 + 4, id="4s wait, 5 cycle steps"),
    ],
)
def test_protoclient_timeout(
    backbone_timeout: float,
    exp_wait_max: float,
    the_backbone: BackboneFeatureStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that attempts to attach to the worker queue from the protoclient
    timeout in an appropriate amount of time. Note: due to the backoff, we verify
    the elapsed time is less than the 15s of a cycle of waits.

    :param backbone_timeout: a timeout for use when configuring a proto client
    :param exp_wait_max: a ceiling for the expected time spent waiting for
    the timeout
    :param the_backbone: a pre-initialized backbone featurestore for setting up
    the environment variable required by the client
    """

    # NOTE: exp_wait_time maps to the cycled backoff of [0.1, 0.2, 0.4, 0.8]
    # with leeway added (by allowing 1s each for the 0.1 and 0.5 steps)

    with monkeypatch.context() as ctx, pytest.raises(SmartSimError) as ex:
        start_time = time.time()
        # remove the worker queue value from the backbone if it exists
        # to ensure the timeout occurs
        the_backbone.pop(BackboneFeatureStore.MLI_WORKER_QUEUE)

        ctx.setenv(BackboneFeatureStore.MLI_BACKBONE, the_backbone.descriptor)

        ProtoClient(timing_on=False, backbone_timeout=backbone_timeout)
        elapsed = time.time() - start_time
        logger.info(f"ProtoClient timeout occurred in {elapsed} seconds")

        # confirm that we met our timeout
        assert (
            elapsed >= backbone_timeout
        ), f"below configured timeout {backbone_timeout}"

        # confirm that the total wait time is aligned with the sleep cycle
        assert elapsed < exp_wait_max, f"above expected max wait {exp_wait_max}"


def test_protoclient_initialization_no_backbone(
    monkeypatch: pytest.MonkeyPatch, the_worker_queue: DragonFLIChannel
):
    """Verify that attempting to start the client without required environment variables
    results in an exception.

    :param the_worker_queue: Passing the worker queue fixture to ensure
    the worker queue environment is correctly configured.

    NOTE: os.environ[BackboneFeatureStore.MLI_BACKBONE] is not set"""

    with monkeypatch.context() as patch, pytest.raises(SmartSimError) as ex:
        patch.setenv(BackboneFeatureStore.MLI_BACKBONE, "")

        ProtoClient(timing_on=False)

    # confirm the missing value error has been raised
    assert {"backbone", "configuration"}.issubset(set(ex.value.args[0].split(" ")))


def test_protoclient_initialization(
    the_backbone: BackboneFeatureStore,
    the_worker_queue: DragonFLIChannel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that attempting to start the client with required env vars results
    in a fully initialized client.

    :param the_backbone: a pre-initialized backbone featurestore
    :param the_worker_queue: an FLI channel the client will retrieve
    from the backbone"""

    with monkeypatch.context() as ctx:
        ctx.setenv(BackboneFeatureStore.MLI_BACKBONE, the_backbone.descriptor)
        # NOTE: rely on `the_worker_queue` fixture to put MLI_WORKER_QUEUE in backbone

        client = ProtoClient(timing_on=False)

        fs_descriptor = the_backbone.descriptor
        wq_descriptor = the_worker_queue.descriptor

        # confirm the backbone was attached correctly
        assert client._backbone is not None
        assert client._backbone.descriptor == fs_descriptor

        # we expect the backbone to add its descriptor to the local env
        assert os.environ[BackboneFeatureStore.MLI_BACKBONE] == fs_descriptor

        # confirm the worker queue is created and attached correctly
        assert client._to_worker_fli is not None
        assert client._to_worker_fli.descriptor == wq_descriptor

        # we expect the worker queue descriptor to be placed into the backbone
        # we do NOT expect _from_worker_ch to be placed anywhere. it's a specific callback
        assert the_backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] == wq_descriptor

        # confirm the worker channels are created
        assert client._from_worker_ch is not None
        assert client._to_worker_ch is not None

        # wrap the channels just to easily verify they produces a descriptor
        assert DragonCommChannel(client._from_worker_ch.channel).descriptor
        assert DragonCommChannel(client._to_worker_ch).descriptor

        # confirm a publisher is created
        assert client._publisher is not None


def test_protoclient_write_model(
    the_backbone: BackboneFeatureStore,
    the_worker_queue: DragonFLIChannel,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that writing a model using the client causes the model data to be
    written to a feature store.

    :param the_backbone: a pre-initialized backbone featurestore
    :param the_worker_queue: Passing the worker queue fixture to ensure
    the worker queue environment is correctly configured.
    from the backbone
    """

    with monkeypatch.context() as ctx:
        # we won't actually send here
        client = ProtoClient(timing_on=False)

        ctx.setenv(BackboneFeatureStore.MLI_BACKBONE, the_backbone.descriptor)
        # NOTE: rely on `the_worker_queue` fixture to put MLI_WORKER_QUEUE in backbone

        client = ProtoClient(timing_on=False)

        model_key = "my-model"
        model_bytes = b"12345"

        client.set_model(model_key, model_bytes)

        # confirm the client modified the underlying feature store
        assert client._backbone[model_key] == model_bytes


@pytest.mark.parametrize(
    "num_listeners, num_model_updates",
    [(1, 1), (1, 4), (2, 4), (16, 4), (64, 8)],
)
def test_protoclient_write_model_notification_sent(
    the_backbone: BackboneFeatureStore,
    the_worker_queue: DragonFLIChannel,
    monkeypatch: pytest.MonkeyPatch,
    num_listeners: int,
    num_model_updates: int,
):
    """Verify that writing a model sends a key-written event.

    :param the_backbone: a pre-initialized backbone featurestore
    :param the_worker_queue: an FLI channel the client will retrieve
    from the backbone
    :param num_listeners: vary the number of registered listeners
    to verify that the event is broadcast to everyone
    :param num_listeners: vary the number of listeners to register
    to verify the broadcast counts messages sent correctly
    """

    # we won't actually send here, but it won't try without registered listeners
    listeners = [f"mock-ch-desc-{i}" for i in range(num_listeners)]

    the_backbone[BackboneFeatureStore.MLI_BACKBONE] = the_backbone.descriptor
    the_backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] = the_worker_queue.descriptor
    the_backbone[BackboneFeatureStore.MLI_NOTIFY_CONSUMERS] = ",".join(listeners)
    the_backbone[BackboneFeatureStore.MLI_REGISTRAR_CONSUMER] = None

    with monkeypatch.context() as ctx:
        ctx.setenv(BackboneFeatureStore.MLI_BACKBONE, the_backbone.descriptor)
        # NOTE: rely on `the_worker_queue` fixture to put MLI_WORKER_QUEUE in backbone

        client = ProtoClient(timing_on=False)

        publisher = t.cast(EventBroadcaster, client._publisher)

        # mock attaching to a channel given the mock-ch-desc in backbone
        mock_send = MagicMock(return_value=None)
        mock_comm_channel = MagicMock(**{"send": mock_send}, spec=DragonCommChannel)
        mock_get_comm_channel = MagicMock(return_value=mock_comm_channel)
        ctx.setattr(publisher, "_get_comm_channel", mock_get_comm_channel)

        model_key = "my-model"
        model_bytes = b"12345"

        for i in range(num_model_updates):
            client.set_model(model_key, model_bytes)

        # confirm that a listener channel was attached
        # once for each registered listener in backbone
        assert mock_get_comm_channel.call_count == num_listeners * num_model_updates

        # confirm the client raised the key-written event
        assert (
            mock_send.call_count == num_listeners * num_model_updates
        ), f"Expected {num_listeners} sends with {num_listeners} registrations"

        # with at least 1 consumer registered, we can verify the message is sent
        for call_args in mock_send.call_args_list:
            send_args = call_args.args
            event_bytes, timeout = send_args[0], send_args[1]

            assert event_bytes, "Expected event bytes to be supplied to send"
            assert (
                timeout == 0.001
            ), "Expected default timeout on call to `publisher.send`, "

            # confirm the correct event was raised
            event = t.cast(
                OnWriteFeatureStore,
                pickle.loads(event_bytes),
            )
            assert event.descriptor == the_backbone.descriptor
            assert event.key == model_key
