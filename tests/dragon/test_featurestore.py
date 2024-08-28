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

import typing as t

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.comm.channel.dragonchannel import DragonCommChannel
from smartsim._core.mli.infrastructure.storage.backbonefeaturestore import (
    BackboneFeatureStore,
    EventBroadcaster,
    EventCategory,
    EventConsumer,
    OnCreateConsumer,
    OnWriteFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import dragon_ddict

# isort: off
from dragon.channels import Channel

# isort: on

if t.TYPE_CHECKING:
    import conftest


# The tests in this file must run in a dragon environment
pytestmark = pytest.mark.dragon


@pytest.fixture
def storage_for_dragon_fs() -> t.Dict[str, str]:
    return dragon_ddict.DDict()


def test_eventconsumer_eventpublisher_integration(
    storage_for_dragon_fs: t.Any, test_dir: str
) -> None:
    """Verify that the publisher and consumer integrate as expected when
    multiple publishers and consumers are sending simultaneously. This
    test closely tracks the test in tests/test_featurestore.py also named
    test_eventconsumer_eventpublisher_integration but requires dragon entities

    :param storage_for_dragon_fs: the dragon storage engine to use
    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""

    mock_storage = storage_for_dragon_fs
    backbone = BackboneFeatureStore(mock_storage)
    mock_fs_descriptor = backbone.descriptor

    # verify ability to write and read from ddict
    backbone["test_dir"] = test_dir
    assert backbone["test_dir"] == test_dir

    wmgr_channel_ = Channel.make_process_local()
    capp_channel_ = Channel.make_process_local()
    back_channel_ = Channel.make_process_local()

    wmgr_channel = DragonCommChannel(wmgr_channel_)
    capp_channel = DragonCommChannel(capp_channel_)
    back_channel = DragonCommChannel(back_channel_)

    wmgr_consumer_descriptor = wmgr_channel.descriptor_string
    capp_consumer_descriptor = capp_channel.descriptor_string
    back_consumer_descriptor = back_channel.descriptor_string

    # create some consumers to receive messages
    wmgr_consumer = EventConsumer(
        wmgr_channel,
        backbone,
        filters=[EventCategory.FEATURE_STORE_WRITTEN],
    )
    capp_consumer = EventConsumer(
        capp_channel,
        backbone,
    )
    back_consumer = EventConsumer(
        back_channel,
        backbone,
        filters=[EventCategory.CONSUMER_CREATED],
    )

    # create some broadcasters to publish messages
    mock_worker_mgr = EventBroadcaster(
        backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )
    mock_client_app = EventBroadcaster(
        backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )

    # register all of the consumers even though the OnCreateConsumer really should
    # trigger its registration. event processing is tested elsewhere.
    backbone.notification_channels = [
        wmgr_consumer_descriptor,
        capp_consumer_descriptor,
        back_consumer_descriptor,
    ]

    # simulate worker manager sending a notification to backend that it's alive
    event_1 = OnCreateConsumer(wmgr_consumer_descriptor)
    mock_worker_mgr.send(event_1)

    # simulate the app updating a model a few times
    event_2 = OnWriteFeatureStore(backbone.descriptor, "key-1")
    event_3 = OnWriteFeatureStore(backbone.descriptor, "key-2")
    event_4 = OnWriteFeatureStore(backbone.descriptor, "key-1")

    mock_client_app.send(event_2)
    mock_client_app.send(event_3)
    mock_client_app.send(event_4)

    # worker manager should only get updates about feature update
    wmgr_messages = wmgr_consumer.receive()
    assert len(wmgr_messages) == 3

    # the backend should only receive messages about consumer creation
    back_messages = back_consumer.receive()
    assert len(back_messages) == 1

    # hypothetical app has no filters and will get all events
    app_messages = capp_consumer.receive()
    assert len(app_messages) == 4
