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

from smartsim._core.mli.comm.channel.dragon_channel import (
    DEFAULT_CHANNEL_BUFFER_SIZE,
    DragonCommChannel,
)
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBroadcaster,
    EventCategory,
    EventConsumer,
    OnCreateConsumer,
    OnWriteFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import dragon_ddict

# isort: off
from dragon.channels import Channel

# isort: on

if t.TYPE_CHECKING:
    import conftest


# The tests in this file must run in a dragon environment
pytestmark = pytest.mark.dragon


@pytest.fixture(scope="module")
def storage_for_dragon_fs() -> t.Dict[str, str]:
    """Fixture to instantiate a dragon distributed dictionary.

    NOTE: using module scoped fixtures drastically improves test run-time"""
    return dragon_ddict.DDict(1, 2, total_mem=2 * 1024**3)


@pytest.fixture(scope="module")
def the_worker_channel() -> DragonCommChannel:
    """Fixture to create a valid descriptor for a worker channel
    that can be attached to.

    NOTE: using module scoped fixtures drastically improves test run-time"""
    wmgr_channel_ = create_local()
    wmgr_channel = DragonCommChannel(wmgr_channel_)
    return wmgr_channel


@pytest.fixture(scope="module")
def the_backbone(storage_for_dragon_fs: t.Any) -> BackboneFeatureStore:
    """Fixture to create a distributed dragon dictionary and wrap it
    in a BackboneFeatureStore.

    NOTE: using module scoped fixtures drastically improves test run-time"""
    return BackboneFeatureStore(storage_for_dragon_fs, allow_reserved_writes=True)


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
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)

    # verify ability to write and read from ddict
    backbone["test_dir"] = test_dir
    assert backbone["test_dir"] == test_dir

    wmgr_channel = DragonCommChannel(create_local())
    capp_channel = DragonCommChannel(create_local())
    back_channel = DragonCommChannel(create_local())

    wmgr_consumer_descriptor = wmgr_channel.descriptor
    capp_consumer_descriptor = capp_channel.descriptor
    back_consumer_descriptor = back_channel.descriptor

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
    event_1 = OnCreateConsumer(wmgr_consumer_descriptor, filters=[])
    mock_worker_mgr.send(event_1)

    # simulate the app updating a model a few times
    for key in ["key-1", "key-2", "key-1"]:
        event = OnWriteFeatureStore(backbone.descriptor, key)
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
    "num_events, batch_timeout, max_batches_expected",
    [
        pytest.param(1, 1.0, 2, id="under 1s timeout"),
        pytest.param(20, 1.0, 3, id="test 1s timeout 20x"),
        pytest.param(30, 0.2, 5, id="test 0.2s timeout 30x"),
        pytest.param(60, 0.4, 4, id="small batches"),
        pytest.param(100, 0.1, 10, id="many small batches"),
    ],
)
def test_eventconsumer_max_dequeue(
    num_events: int,
    batch_timeout: float,
    max_batches_expected: int,
    the_worker_channel: DragonCommChannel,
    the_backbone: BackboneFeatureStore,
) -> None:
    """Verify that a consumer does not sit and collect messages indefinitely
    by checking that a consumer returns after a maximum timeout is exceeded.

    :param num_events: Total number of events to raise in the test
    :param batch_timeout: Maximum wait time (in seconds) for a message to be sent
    :param max_batches_expected: Maximum number of receives that should occur
    :param storage_for_dragon_fs: Dragon storage engine to use
    """

    # create some consumers to receive messages
    wmgr_consumer = EventConsumer(
        the_worker_channel,
        the_backbone,
        filters=[EventCategory.FEATURE_STORE_WRITTEN],
    )

    # create a broadcaster to publish messages
    mock_client_app = EventBroadcaster(
        the_backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )

    # register all of the consumers even though the OnCreateConsumer really should
    # trigger its registration. event processing is tested elsewhere.
    the_backbone.notification_channels = [the_worker_channel.descriptor]

    # simulate the app updating a model a lot of times
    for key in (f"key-{i}" for i in range(num_events)):
        event = OnWriteFeatureStore(the_backbone.descriptor, key)
        mock_client_app.send(event, timeout=0.01)

    num_dequeued = 0
    num_batches = 0

    while wmgr_messages := wmgr_consumer.recv(
        timeout=0.1,
        batch_timeout=batch_timeout,
    ):
        # worker manager should not get more than `max_num_msgs` events
        num_dequeued += len(wmgr_messages)
        num_batches += 1

    # make sure we made all the expected dequeue calls and got everything
    assert num_dequeued == num_events
    assert num_batches > 0
    assert num_batches < max_batches_expected, "too many recv calls were made"


@pytest.mark.parametrize(
    "buffer_size",
    [
        pytest.param(
            -1,
            id="replace negative, default to 500",
            marks=pytest.mark.skip("create_local issue w/MPI must be mitigated"),
        ),
        pytest.param(
            0,
            id="replace zero, default to 500",
            marks=pytest.mark.skip("create_local issue w/MPI must be mitigated"),
        ),
        pytest.param(
            1,
            id="non-zero buffer size: 1",
            marks=pytest.mark.skip("create_local issue w/MPI must be mitigated"),
        ),
        # pytest.param(500, id="maximum size edge case: 500"),
        pytest.param(
            550,
            id="larger than default: 550",
            marks=pytest.mark.skip("create_local issue w/MPI must be mitigated"),
        ),
        pytest.param(
            800,
            id="much larger then default: 800",
            marks=pytest.mark.skip("create_local issue w/MPI must be mitigated"),
        ),
        pytest.param(
            1000,
            id="very large buffer: 1000, unreliable in dragon-v0.10",
            marks=pytest.mark.skip("create_local issue w/MPI must be mitigated"),
        ),
    ],
)
def test_channel_buffer_size(
    buffer_size: int,
    storage_for_dragon_fs: t.Any,
) -> None:
    """Verify that a channel used by an EventBroadcaster can buffer messages
    until a configured maximum value is exceeded.

    :param buffer_size: the maximum number of messages allowed in a channel buffer
    :param storage_for_dragon_fs: the dragon storage engine to use"""

    mock_storage = storage_for_dragon_fs
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)

    wmgr_channel_ = create_local(buffer_size)  # <--- vary buffer size
    wmgr_channel = DragonCommChannel(wmgr_channel_)
    wmgr_consumer_descriptor = wmgr_channel.descriptor

    # create a broadcaster to publish messages. create no consumers to
    # push the number of sent messages past the allotted buffer size
    mock_client_app = EventBroadcaster(
        backbone,
        channel_factory=DragonCommChannel.from_descriptor,
    )

    # register all of the consumers even though the OnCreateConsumer really should
    # trigger its registration. event processing is tested elsewhere.
    backbone.notification_channels = [wmgr_consumer_descriptor]

    if buffer_size < 1:
        # NOTE: we set this after creating the channel above to ensure
        # the default parameter value was used during instantiation
        buffer_size = DEFAULT_CHANNEL_BUFFER_SIZE

    # simulate the app updating a model a lot of times
    for key in (f"key-{i}" for i in range(buffer_size)):
        event = OnWriteFeatureStore(backbone.descriptor, key)
        mock_client_app.send(event, timeout=0.01)

    # adding 1 more over the configured buffer size should report the error
    with pytest.raises(Exception) as ex:
        mock_client_app.send(event, timeout=0.01)
