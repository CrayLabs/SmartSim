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

import pytest

dragon = pytest.importorskip("dragon")


from smartsim._core.launcher.dragon.dragonBackend import DragonBackend
from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBase,
    EventBroadcaster,
    EventConsumer,
    OnCreateConsumer,
)
from smartsim.log import get_logger

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon
logger = get_logger(__name__)


def test_dragonbackend_listener_boostrapping(monkeypatch: pytest.MonkeyPatch):
    """Verify that the dragon backend registration channel correctly
    registers new consumers in the backbone and begins sending events
    to the new consumers."""

    backend = DragonBackend(pid=9999)

    backend._create_backbone()
    backbone = backend._backbone

    def mock_event_handler(event: EventBase) -> None:
        logger.debug(f"Handling event in mock handler: {event}")

        bb_descriptor = os.environ.get(BackboneFeatureStore.MLI_BACKBONE, None)
        assert bb_descriptor

        fs = BackboneFeatureStore.from_descriptor(bb_descriptor)
        fs[event.uid] = "received"

    # create the consumer and start a listener process
    backend_consumer = backend._create_eventing(backbone)
    registrar_descriptor = backend._event_consumer.descriptor

    # ensure the consumer is stored to backend & published to backbone
    assert backend._event_consumer == backend_consumer
    assert backbone.backend_channel == registrar_descriptor
    assert os.environ.get(BackboneFeatureStore.MLI_BACKBONE, None)

    # simulate a new consumer registration
    new_consumer_ch = DragonCommChannel.from_local()
    new_consumer = EventConsumer(
        new_consumer_ch,
        backbone,
        [],
        name="test-consumer-a",
        event_handler=mock_event_handler,
    )
    assert new_consumer, "new_consumer construction failed"

    # send registration to registrar channel
    new_consumer.register()

    # the backend consumer should handle updating the notify list and the new
    # consumer that just broadcast its registration should be registered...
    # backend_consumer.listen_once(timeout=2.0)
    backend.listen_to_registrations(timeout=0.1)

    # # confirm the backend registrar consumer registerd the new listener
    assert new_consumer_ch.descriptor in backbone.notification_channels

    broadcaster = EventBroadcaster(backbone, DragonCommChannel.from_descriptor)

    # re-send the same thing because i'm too lazy to create a new consumer
    broadcast_event = OnCreateConsumer(registrar_descriptor, [])
    broadcaster.send(broadcast_event, timeout=0.1)

    new_consumer.listen_once(timeout=0.1)

    values = backbone.wait_for(
        [broadcast_event.uid, BackboneFeatureStore.MLI_NOTIFY_CONSUMERS], 1.0
    )
    stored = values[broadcast_event.uid]
    assert stored == "received", "The handler didn't update the backbone"

    # confirm that directly retrieving the value isn't different from
    # using backbone.notification_channels helper method
    notify_list = str(values[BackboneFeatureStore.MLI_NOTIFY_CONSUMERS]).split(",")
    assert new_consumer.descriptor in set(notify_list)
