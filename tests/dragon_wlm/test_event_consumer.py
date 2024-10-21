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

import time
import typing as t
from unittest import mock

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.comm.broadcaster import EventBroadcaster
from smartsim._core.mli.infrastructure.comm.consumer import EventConsumer
from smartsim._core.mli.infrastructure.comm.event import (
    OnCreateConsumer,
    OnShutdownRequested,
    OnWriteFeatureStore,
)
from smartsim._core.mli.infrastructure.control.listener import (
    ConsumerRegistrationListener,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim.log import get_logger

logger = get_logger(__name__)

# isort: off
from dragon import fli
from dragon.channels import Channel

# isort: on

if t.TYPE_CHECKING:
    import conftest


# The tests in this file must run in a dragon environment
pytestmark = pytest.mark.dragon


def test_eventconsumer_eventpublisher_integration(
    the_backbone: t.Any, test_dir: str
) -> None:
    """Verify that the publisher and consumer integrate as expected when
    multiple publishers and consumers are sending simultaneously. This
    test closely tracks the test in tests/test_featurestore_base.py also named
    test_eventconsumer_eventpublisher_integration but requires dragon entities.

    :param the_backbone: The BackboneFeatureStore to use
    :param test_dir: Automatically generated unique working
    directories for individual test outputs
    """

    wmgr_channel = DragonCommChannel(create_local())
    capp_channel = DragonCommChannel(create_local())
    back_channel = DragonCommChannel(create_local())

    wmgr_consumer_descriptor = wmgr_channel.descriptor
    capp_consumer_descriptor = capp_channel.descriptor
    back_consumer_descriptor = back_channel.descriptor

    # create some consumers to receive messages
    wmgr_consumer = EventConsumer(
        wmgr_channel,
        the_backbone,
        filters=[OnWriteFeatureStore.FEATURE_STORE_WRITTEN],
    )
    capp_consumer = EventConsumer(
        capp_channel,
        the_backbone,
    )
    back_consumer = EventConsumer(
        back_channel,
        the_backbone,
        filters=[OnCreateConsumer.CONSUMER_CREATED],
    )

    # create some broadcasters to publish messages
    mock_worker_mgr = EventBroadcaster(
        the_backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )
    mock_client_app = EventBroadcaster(
        the_backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )

    # register all of the consumers even though the OnCreateConsumer really should
    # trigger its registration. event processing is tested elsewhere.
    the_backbone.notification_channels = [
        wmgr_consumer_descriptor,
        capp_consumer_descriptor,
        back_consumer_descriptor,
    ]

    # simulate worker manager sending a notification to backend that it's alive
    event_1 = OnCreateConsumer(
        "test_eventconsumer_eventpublisher_integration",
        wmgr_consumer_descriptor,
        filters=[],
    )
    mock_worker_mgr.send(event_1)

    # simulate the app updating a model a few times
    for key in ["key-1", "key-2", "key-1"]:
        event = OnWriteFeatureStore(
            "test_eventconsumer_eventpublisher_integration",
            the_backbone.descriptor,
            key,
        )
        mock_client_app.send(event, timeout=0.1)

    # worker manager should only get updates about feature update
    wmgr_messages = wmgr_consumer.recv()
    assert len(wmgr_messages) == 3

    # the backend should only receive messages about consumer creation
    back_messages = back_consumer.recv()
    assert len(back_messages) == 1

    # hypothetical app has no filters and will get all events
    app_messages = capp_consumer.recv()
    assert len(app_messages) == 4


@pytest.mark.parametrize(
    " timeout, batch_timeout, exp_err_msg",
    [(-1, 1, " timeout"), (1, -1, "batch_timeout")],
)
def test_eventconsumer_invalid_timeout(
    timeout: float,
    batch_timeout: float,
    exp_err_msg: str,
    test_dir: str,
    the_backbone: BackboneFeatureStore,
) -> None:
    """Verify that the event consumer raises an exception
    when provided an invalid request timeout.

    :param timeout: The request timeout for the event consumer recv call
    :param batch_timeout: The batch timeout for the event consumer recv call
    :param exp_err_msg: A unique value from the error message that should be raised
    :param the_storage: The dragon storage engine to use
    :param test_dir: Automatically generated unique working
    directories for individual test outputs
    """

    wmgr_channel = DragonCommChannel(create_local())

    # create some consumers to receive messages
    wmgr_consumer = EventConsumer(
        wmgr_channel,
        the_backbone,
        filters=[OnWriteFeatureStore.FEATURE_STORE_WRITTEN],
    )

    # the consumer should report an error for the invalid timeout value
    with pytest.raises(ValueError) as ex:
        wmgr_consumer.recv(timeout=timeout, batch_timeout=batch_timeout)

    assert exp_err_msg in ex.value.args[0]


def test_eventconsumer_no_event_handler_registered(
    the_backbone: t.Any, test_dir: str
) -> None:
    """Verify that a consumer discards messages when
    on a channel if no handler is registered.

    :param the_backbone: The BackboneFeatureStore to use
    :param test_dir: Automatically generated unique working
    directories for individual test outputs
    """

    wmgr_channel = DragonCommChannel(create_local())

    # create a consumer to receive messages
    wmgr_consumer = EventConsumer(wmgr_channel, the_backbone, event_handler=None)

    # create a broadcasters to publish messages
    mock_worker_mgr = EventBroadcaster(
        the_backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )

    # manually register the consumers since we don't have a backend running
    the_backbone.notification_channels = [wmgr_channel.descriptor]

    # simulate the app updating a model a few times
    for key in ["key-1", "key-2", "key-1"]:
        event = OnWriteFeatureStore(
            "test_eventconsumer_no_event_handler_registered",
            the_backbone.descriptor,
            key,
        )
        mock_worker_mgr.send(event, timeout=0.1)

    # run the handler and let it discard messages
    for _ in range(15):
        wmgr_consumer.listen_once(0.2, 2.0)

    assert wmgr_consumer.listening


def test_eventconsumer_no_event_handler_registered_shutdown(
    the_backbone: t.Any, test_dir: str
) -> None:
    """Verify that a consumer without an event handler
    registered still honors shutdown requests.

    :param the_backbone: The BackboneFeatureStore to use
    :param test_dir: Automatically generated unique working
    directories for individual test outputs
    """

    wmgr_channel = DragonCommChannel(create_local())
    capp_channel = DragonCommChannel(create_local())

    # create a consumers to receive messages
    wmgr_consumer = EventConsumer(wmgr_channel, the_backbone)

    # create a broadcaster to publish messages
    mock_worker_mgr = EventBroadcaster(
        the_backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )

    # manually register the consumers since we don't have a backend running
    the_backbone.notification_channels = [
        wmgr_channel.descriptor,
        capp_channel.descriptor,
    ]

    # simulate the app updating a model a few times
    for key in ["key-1", "key-2", "key-1"]:
        event = OnWriteFeatureStore(
            "test_eventconsumer_no_event_handler_registered_shutdown",
            the_backbone.descriptor,
            key,
        )
        mock_worker_mgr.send(event, timeout=0.1)

    event = OnShutdownRequested(
        "test_eventconsumer_no_event_handler_registered_shutdown"
    )
    mock_worker_mgr.send(event, timeout=0.1)

    # wmgr will stop listening to messages when it is told to stop listening
    wmgr_consumer.listen(timeout=0.1, batch_timeout=2.0)

    for _ in range(15):
        wmgr_consumer.listen_once(timeout=0.1, batch_timeout=2.0)

    # confirm the messages were processed, discarded, and the shutdown was received
    assert wmgr_consumer.listening == False


def test_eventconsumer_registration(
    the_backbone: t.Any, test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that a consumer is correctly registered in
    the backbone after sending a registration request. Then,
    Confirm the consumer is unregistered after sending the
    un-register request.

    :param the_backbone: The BackboneFeatureStore to use
    :param test_dir: Automatically generated unique working
    directories for individual test outputs
    """

    with monkeypatch.context() as patch:
        registrar = ConsumerRegistrationListener(
            the_backbone, 1.0, 2.0, as_service=False
        )

        # NOTE: service.execute(as_service=False) will complete the service life-
        # cycle and remove the registrar from the backbone, so mock _on_shutdown
        disabled_shutdown = mock.MagicMock()
        patch.setattr(registrar, "_on_shutdown", disabled_shutdown)

        # initialze registrar resources
        registrar.execute()

        # create a consumer that will be registered
        wmgr_channel = DragonCommChannel(create_local())
        wmgr_consumer = EventConsumer(wmgr_channel, the_backbone)

        registered_channels = the_backbone.notification_channels

        # trigger the consumer-to-registrar handshake
        wmgr_consumer.register()

        current_registrations: t.List[str] = []

        # have the registrar run a few times to pick up the msg
        for i in range(15):
            registrar.execute()
            current_registrations = the_backbone.notification_channels
            if len(current_registrations) != len(registered_channels):
                logger.debug(f"The event was processed on iteration {i}")
                break

        # confirm the consumer is registered
        assert wmgr_channel.descriptor in current_registrations

        # copy old list so we can compare against it.
        registered_channels = list(current_registrations)

        # trigger the consumer removal
        wmgr_consumer.unregister()

        # have the registrar run a few times to pick up the msg
        for i in range(15):
            registrar.execute()
            current_registrations = the_backbone.notification_channels
            if len(current_registrations) != len(registered_channels):
                logger.debug(f"The event was processed on iteration {i}")
                break

        # confirm the consumer is no longer registered
        assert wmgr_channel.descriptor not in current_registrations


def test_registrar_teardown(
    the_backbone: t.Any, test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that the consumer registrar removes itself from
    the backbone when it shuts down.

    :param the_backbone: The BackboneFeatureStore to use
    :param test_dir: Automatically generated unique working
    directories for individual test outputs
    """

    with monkeypatch.context() as patch:
        registrar = ConsumerRegistrationListener(
            the_backbone, 1.0, 2.0, as_service=False
        )

        # directly initialze registrar resources to avoid service life-cycle
        registrar._create_eventing()

        # confirm the registrar is published to the backbone
        cfg = the_backbone.wait_for([BackboneFeatureStore.MLI_REGISTRAR_CONSUMER], 10)
        assert BackboneFeatureStore.MLI_REGISTRAR_CONSUMER in cfg

        # execute the entire service lifecycle 1x
        registrar.execute()

        consumer_found = BackboneFeatureStore.MLI_REGISTRAR_CONSUMER in the_backbone

        for i in range(15):
            time.sleep(0.1)
            consumer_found = BackboneFeatureStore.MLI_REGISTRAR_CONSUMER in the_backbone
            if not consumer_found:
                logger.debug(f"Registrar removed from the backbone on iteration {i}")
                break

        assert BackboneFeatureStore.MLI_REGISTRAR_CONSUMER not in the_backbone
