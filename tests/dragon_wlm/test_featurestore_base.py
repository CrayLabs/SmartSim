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
import pathlib
import time
import typing as t

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.infrastructure.comm.broadcaster import EventBroadcaster
from smartsim._core.mli.infrastructure.comm.consumer import EventConsumer
from smartsim._core.mli.infrastructure.comm.event import (
    OnCreateConsumer,
    OnWriteFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.feature_store import ReservedKeys
from smartsim.error import SmartSimError

from .channel import FileSystemCommChannel
from .feature_store import MemoryFeatureStore

if t.TYPE_CHECKING:
    import conftest


# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon


def boom(*args, **kwargs) -> None:
    """Helper function that blows up when used to mock up
    some other function."""
    raise Exception(f"you shall not pass! {args}, {kwargs}")


def test_event_uid() -> None:
    """Verify that all events include a unique identifier."""
    uids: t.Set[str] = set()
    num_iters = 1000

    # generate a bunch of events and keep track all the IDs
    for i in range(num_iters):
        event_a = OnCreateConsumer("test_event_uid", str(i), filters=[])
        event_b = OnWriteFeatureStore("test_event_uid", "test_event_uid", str(i))

        uids.add(event_a.uid)
        uids.add(event_b.uid)

    # verify each event created a unique ID
    assert len(uids) == 2 * num_iters


def test_mli_reserved_keys_conversion() -> None:
    """Verify that conversion from a string to an enum member
    works as expected."""

    for reserved_key in ReservedKeys:
        # iterate through all keys and verify `from_string` works
        assert ReservedKeys.contains(reserved_key.value)

        # show that the value (actual key) not the enum member name
        # will not be incorrectly identified as reserved
        assert not ReservedKeys.contains(str(reserved_key).split(".")[1])


def test_mli_reserved_keys_writes() -> None:
    """Verify that attempts to write to reserved keys are blocked from a
    standard DragonFeatureStore but enabled with the BackboneFeatureStore."""

    mock_storage = {}
    dfs = DragonFeatureStore(mock_storage)
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    other = MemoryFeatureStore(mock_storage)

    expected_value = "value"

    for reserved_key in ReservedKeys:
        # we expect every reserved key to fail using DragonFeatureStore...
        with pytest.raises(SmartSimError) as ex:
            dfs[reserved_key] = expected_value

        assert "reserved key" in ex.value.args[0]

        # ... and expect other feature stores to respect reserved keys
        with pytest.raises(SmartSimError) as ex:
            other[reserved_key] = expected_value

        assert "reserved key" in ex.value.args[0]

        # ...and those same keys to succeed on the backbone
        backbone[reserved_key] = expected_value
        actual_value = backbone[reserved_key]
        assert actual_value == expected_value


def test_mli_consumers_read_by_key() -> None:
    """Verify that the value returned from the mli consumers method is written
    to the correct key and reads are allowed via standard dragon feature store."""

    mock_storage = {}
    dfs = DragonFeatureStore(mock_storage)
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    other = MemoryFeatureStore(mock_storage)

    expected_value = "value"

    # write using backbone that has permission to write reserved keys
    backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS] = expected_value

    # confirm read-only access to reserved keys from any FeatureStore
    for fs in [dfs, backbone, other]:
        assert fs[ReservedKeys.MLI_NOTIFY_CONSUMERS] == expected_value


def test_mli_consumers_read_by_backbone() -> None:
    """Verify that the backbone reads the correct location
    when using the backbone feature store API instead of mapping API."""

    mock_storage = {}
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    expected_value = "value"

    backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS] = expected_value

    # confirm reading via convenience method returns expected value
    assert backbone.notification_channels[0] == expected_value


def test_mli_consumers_write_by_backbone() -> None:
    """Verify that the backbone writes the correct location
    when using the backbone feature store API instead of mapping API."""

    mock_storage = {}
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    expected_value = ["value"]

    backbone.notification_channels = expected_value

    # confirm write using convenience method targets expected key
    assert backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS] == ",".join(expected_value)


def test_eventpublisher_broadcast_no_factory(test_dir: str) -> None:
    """Verify that a broadcast operation without any registered subscribers
    succeeds without raising Exceptions.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}
    consumer_descriptor = storage_path / "test-consumer"

    # NOTE: we're not putting any consumers into the backbone here!
    backbone = BackboneFeatureStore(mock_storage)

    event = OnCreateConsumer(
        "test_eventpublisher_broadcast_no_factory", consumer_descriptor, filters=[]
    )

    publisher = EventBroadcaster(backbone)
    num_receivers = 0

    # publishing this event without any known consumers registered should succeed
    # but report that it didn't have anybody to send the event to
    consumer_descriptor = storage_path / f"test-consumer"
    event = OnCreateConsumer(
        "test_eventpublisher_broadcast_no_factory", consumer_descriptor, filters=[]
    )

    num_receivers += publisher.send(event)

    # confirm no changes to the backbone occur when fetching the empty consumer key
    key_in_features_store = ReservedKeys.MLI_NOTIFY_CONSUMERS in backbone
    assert not key_in_features_store

    # confirm that the broadcast reports no events published
    assert num_receivers == 0
    # confirm that the broadcast buffered the event for a later send
    assert publisher.num_buffered == 1


def test_eventpublisher_broadcast_to_empty_consumer_list(test_dir: str) -> None:
    """Verify that a broadcast operation without any registered subscribers
    succeeds without raising Exceptions.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    # prep our backbone with a consumer list
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    backbone.notification_channels = []

    event = OnCreateConsumer(
        "test_eventpublisher_broadcast_to_empty_consumer_list",
        consumer_descriptor,
        filters=[],
    )
    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )
    num_receivers = publisher.send(event)

    registered_consumers = backbone[ReservedKeys.MLI_NOTIFY_CONSUMERS]

    # confirm that no consumers exist in backbone to send to
    assert not registered_consumers
    # confirm that the broadcast reports no events published
    assert num_receivers == 0
    # confirm that the broadcast buffered the event for a later send
    assert publisher.num_buffered == 1


def test_eventpublisher_broadcast_without_channel_factory(test_dir: str) -> None:
    """Verify that a broadcast operation reports an error if no channel
    factory was supplied for constructing the consumer channels.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    # prep our backbone with a consumer list
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    backbone.notification_channels = [consumer_descriptor]

    event = OnCreateConsumer(
        "test_eventpublisher_broadcast_without_channel_factory",
        consumer_descriptor,
        filters=[],
    )
    publisher = EventBroadcaster(
        backbone,
        # channel_factory=FileSystemCommChannel.from_descriptor # <--- not supplied
    )

    with pytest.raises(SmartSimError) as ex:
        publisher.send(event)

    assert "factory" in ex.value.args[0]


def test_eventpublisher_broadcast_empties_buffer(test_dir: str) -> None:
    """Verify that a successful broadcast clears messages from the event
    buffer when a new message is sent and consumers are registered.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    backbone.notification_channels = (consumer_descriptor,)

    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    # mock building up some buffered events
    num_buffered_events = 14
    for i in range(num_buffered_events):
        event = OnCreateConsumer(
            "test_eventpublisher_broadcast_empties_buffer",
            storage_path / f"test-consumer-{str(i)}",
            [],
        )
        publisher._event_buffer.append(bytes(event))

    event0 = OnCreateConsumer(
        "test_eventpublisher_broadcast_empties_buffer",
        storage_path / f"test-consumer-{str(num_buffered_events + 1)}",
        [],
    )

    num_receivers = publisher.send(event0)
    # 1 receiver x 15 total events == 15 events
    assert num_receivers == num_buffered_events + 1


@pytest.mark.parametrize(
    "num_consumers, num_buffered, expected_num_sent",
    [
        pytest.param(0, 7, 0, id="0 x (7+1) - no consumers, multi-buffer"),
        pytest.param(1, 7, 8, id="1 x (7+1) - single consumer, multi-buffer"),
        pytest.param(2, 7, 16, id="2 x (7+1) - multi-consumer, multi-buffer"),
        pytest.param(4, 4, 20, id="4 x (4+1) - multi-consumer, multi-buffer (odd #)"),
        pytest.param(9, 0, 9, id="13 x (0+1) - multi-consumer, empty buffer"),
    ],
)
def test_eventpublisher_broadcast_returns_total_sent(
    test_dir: str, num_consumers: int, num_buffered: int, expected_num_sent: int
) -> None:
    """Verify that a successful broadcast returns the total number of events
    sent, including buffered messages.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param num_consumers: the number of consumers to mock setting up prior to send
    :param num_buffered: the number of pre-buffered events to mock up
    :param expected_num_sent: the expected result from calling send
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumers = []
    for i in range(num_consumers):
        consumers.append(storage_path / f"test-consumer-{i}")

    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    backbone.notification_channels = consumers

    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    # mock building up some buffered events
    for i in range(num_buffered):
        event = OnCreateConsumer(
            "test_eventpublisher_broadcast_returns_total_sent",
            storage_path / f"test-consumer-{str(i)}",
            [],
        )
        publisher._event_buffer.append(bytes(event))

    assert publisher.num_buffered == num_buffered

    # this event will trigger clearing anything already in buffer
    event0 = OnCreateConsumer(
        "test_eventpublisher_broadcast_returns_total_sent",
        storage_path / f"test-consumer-{num_buffered}",
        [],
    )

    # num_receivers should contain a number that computes w/all consumers and all events
    num_receivers = publisher.send(event0)

    assert num_receivers == expected_num_sent


def test_eventpublisher_prune_unused_consumer(test_dir: str) -> None:
    """Verify that any unused consumers are pruned each time a new event is sent.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    mock_storage = {}

    # note: file-system descriptors are just paths
    consumer_descriptor = storage_path / "test-consumer"

    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)

    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    event = OnCreateConsumer(
        "test_eventpublisher_prune_unused_consumer",
        consumer_descriptor,
        filters=[],
    )

    # the only registered cnosumer is in the event, expect no pruning
    backbone.notification_channels = (consumer_descriptor,)

    publisher.send(event)
    assert str(consumer_descriptor) in publisher._channel_cache
    assert len(publisher._channel_cache) == 1

    # add a new descriptor for another event...
    consumer_descriptor2 = storage_path / "test-consumer-2"
    # ... and remove the old descriptor from the backbone when it's looked up
    backbone.notification_channels = (consumer_descriptor2,)

    event = OnCreateConsumer(
        "test_eventpublisher_prune_unused_consumer", consumer_descriptor2, filters=[]
    )

    publisher.send(event)

    assert str(consumer_descriptor2) in publisher._channel_cache
    assert str(consumer_descriptor) not in publisher._channel_cache
    assert len(publisher._channel_cache) == 1

    # test multi-consumer pruning by caching some extra channels
    prune0, prune1, prune2 = "abc", "def", "ghi"
    publisher._channel_cache[prune0] = "doesnt-matter-if-it-is-pruned"
    publisher._channel_cache[prune1] = "doesnt-matter-if-it-is-pruned"
    publisher._channel_cache[prune2] = "doesnt-matter-if-it-is-pruned"

    # add in one of our old channels so we prune the above items, send to these
    backbone.notification_channels = (consumer_descriptor, consumer_descriptor2)

    publisher.send(event)

    assert str(consumer_descriptor2) in publisher._channel_cache

    # NOTE: we should NOT prune something that isn't used by this message but
    # does appear in `backbone.notification_channels`
    assert str(consumer_descriptor) in publisher._channel_cache

    # confirm all of our items that were not in the notification channels are gone
    for pruned in [prune0, prune1, prune2]:
        assert pruned not in publisher._channel_cache

    # confirm we have only the two expected items in the channel cache
    assert len(publisher._channel_cache) == 2


def test_eventpublisher_serialize_failure(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that errors during message serialization are raised to the caller.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param monkeypatch: pytest fixture for modifying behavior of existing code
    with mock implementations
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}

    # note: file-system descriptors are just paths
    target_descriptor = str(storage_path / "test-consumer")

    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    with monkeypatch.context() as patch:
        event = OnCreateConsumer(
            "test_eventpublisher_serialize_failure", target_descriptor, filters=[]
        )

        # patch the __bytes__ implementation to cause pickling to fail during send
        def bad_bytes(self) -> bytes:
            return b"abc"

        # this patch causes an attribute error when event pickling is attempted
        patch.setattr(event, "__bytes__", bad_bytes)

        backbone.notification_channels = (target_descriptor,)

        # send a message into the channel
        with pytest.raises(AttributeError) as ex:
            publisher.send(event)

        assert "serialize" in ex.value.args[0]


def test_eventpublisher_factory_failure(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that errors during channel construction are raised to the caller.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param monkeypatch: pytest fixture for modifying behavior of existing code
    with mock implementations
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}

    # note: file-system descriptors are just paths
    target_descriptor = str(storage_path / "test-consumer")

    def boom(descriptor: str) -> None:
        raise Exception(f"you shall not pass! {descriptor}")

    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    publisher = EventBroadcaster(backbone, channel_factory=boom)

    with monkeypatch.context() as patch:
        event = OnCreateConsumer(
            "test_eventpublisher_factory_failure", target_descriptor, filters=[]
        )

        backbone.notification_channels = (target_descriptor,)

        # send a message into the channel
        with pytest.raises(SmartSimError) as ex:
            publisher.send(event)

        assert "construct" in ex.value.args[0]


def test_eventpublisher_failure(test_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that unexpected errors during message send are caught and wrapped in a
    SmartSimError so they are not propagated directly to the caller.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param monkeypatch: pytest fixture for modifying behavior of existing code
    with mock implementations
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}

    # note: file-system descriptors are just paths
    target_descriptor = str(storage_path / "test-consumer")

    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    publisher = EventBroadcaster(
        backbone, channel_factory=FileSystemCommChannel.from_descriptor
    )

    def boom(self) -> None:
        raise Exception("That was unexpected...")

    with monkeypatch.context() as patch:
        event = OnCreateConsumer(
            "test_eventpublisher_failure", target_descriptor, filters=[]
        )

        # patch the _broadcast implementation to cause send to fail after
        # after the event has been pickled
        patch.setattr(publisher, "_broadcast", boom)

        backbone.notification_channels = (target_descriptor,)

        # Here, we see the exception raised by broadcast that isn't expected
        # is not allowed directly out, and instead is wrapped in SmartSimError
        with pytest.raises(SmartSimError) as ex:
            publisher.send(event)

        assert "unexpected" in ex.value.args[0]


def test_eventconsumer_receive(test_dir: str) -> None:
    """Verify that a consumer retrieves a message from the given channel.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}

    # note: file-system descriptors are just paths
    target_descriptor = str(storage_path / "test-consumer")

    backbone = BackboneFeatureStore(mock_storage)
    comm_channel = FileSystemCommChannel.from_descriptor(target_descriptor)
    event = OnCreateConsumer(
        "test_eventconsumer_receive", target_descriptor, filters=[]
    )

    # simulate a sent event by writing directly to the input comm channel
    comm_channel.send(bytes(event))

    consumer = EventConsumer(comm_channel, backbone)

    all_received: t.List[OnCreateConsumer] = consumer.recv()
    assert len(all_received) == 1

    # verify we received the same event that was raised
    assert all_received[0].category == event.category
    assert all_received[0].descriptor == event.descriptor


@pytest.mark.parametrize("num_sent", [0, 1, 2, 4, 8, 16])
def test_eventconsumer_receive_multi(test_dir: str, num_sent: int) -> None:
    """Verify that a consumer retrieves multiple message from the given channel.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    :param num_sent: parameterized value used to vary the number of events
    that are enqueued and validations are checked at multiple queue sizes
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}

    # note: file-system descriptors are just paths
    target_descriptor = str(storage_path / "test-consumer")

    backbone = BackboneFeatureStore(mock_storage)
    comm_channel = FileSystemCommChannel.from_descriptor(target_descriptor)

    # simulate multiple sent events by writing directly to the input comm channel
    for _ in range(num_sent):
        event = OnCreateConsumer(
            "test_eventconsumer_receive_multi", target_descriptor, filters=[]
        )
        comm_channel.send(bytes(event))

    consumer = EventConsumer(comm_channel, backbone)

    all_received: t.List[OnCreateConsumer] = consumer.recv()
    assert len(all_received) == num_sent


def test_eventconsumer_receive_empty(test_dir: str) -> None:
    """Verify that a consumer receiving an empty message ignores the
    message and continues processing.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}

    # note: file-system descriptors are just paths
    target_descriptor = str(storage_path / "test-consumer")

    backbone = BackboneFeatureStore(mock_storage)
    comm_channel = FileSystemCommChannel.from_descriptor(target_descriptor)

    # simulate a sent event by writing directly to the input comm channel
    comm_channel.send(bytes(b""))

    consumer = EventConsumer(comm_channel, backbone)

    messages = consumer.recv()

    # the messages array should be empty
    assert not messages


def test_eventconsumer_eventpublisher_integration(test_dir: str) -> None:
    """Verify that the publisher and consumer integrate as expected when
    multiple publishers and consumers are sending simultaneously.

    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}
    backbone = BackboneFeatureStore(mock_storage, allow_reserved_writes=True)
    mock_fs_descriptor = str(storage_path / f"mock-feature-store")

    wmgr_channel = FileSystemCommChannel(storage_path / "test-wmgr")
    capp_channel = FileSystemCommChannel(storage_path / "test-capp")
    back_channel = FileSystemCommChannel(storage_path / "test-backend")

    wmgr_consumer_descriptor = wmgr_channel.descriptor
    capp_consumer_descriptor = capp_channel.descriptor
    back_consumer_descriptor = back_channel.descriptor

    # create some consumers to receive messages
    wmgr_consumer = EventConsumer(
        wmgr_channel,
        backbone,
        filters=[OnWriteFeatureStore.FEATURE_STORE_WRITTEN],
    )
    capp_consumer = EventConsumer(
        capp_channel,
        backbone,
    )
    back_consumer = EventConsumer(
        back_channel,
        backbone,
        filters=[OnCreateConsumer.CONSUMER_CREATED],
    )

    # create some broadcasters to publish messages
    mock_worker_mgr = EventBroadcaster(
        backbone,
        channel_factory=FileSystemCommChannel.from_descriptor,
    )
    mock_client_app = EventBroadcaster(
        backbone,
        channel_factory=FileSystemCommChannel.from_descriptor,
    )

    # register all of the consumers even though the OnCreateConsumer really should
    # trigger its registration. event processing is tested elsewhere.
    backbone.notification_channels = [
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
    event_2 = OnWriteFeatureStore(
        "test_eventconsumer_eventpublisher_integration", mock_fs_descriptor, "key-1"
    )
    event_3 = OnWriteFeatureStore(
        "test_eventconsumer_eventpublisher_integration", mock_fs_descriptor, "key-2"
    )
    event_4 = OnWriteFeatureStore(
        "test_eventconsumer_eventpublisher_integration", mock_fs_descriptor, "key-1"
    )

    mock_client_app.send(event_2)
    mock_client_app.send(event_3)
    mock_client_app.send(event_4)

    # worker manager should only get updates about feature update
    wmgr_messages = wmgr_consumer.recv()
    assert len(wmgr_messages) == 3

    # the backend should only receive messages about consumer creation
    back_messages = back_consumer.recv()
    assert len(back_messages) == 1

    # hypothetical app has no filters and will get all events
    app_messages = capp_consumer.recv()
    assert len(app_messages) == 4


@pytest.mark.parametrize("invalid_timeout", [-100.0, -1.0, 0.0])
def test_eventconsumer_batch_timeout(
    invalid_timeout: float,
    test_dir: str,
) -> None:
    """Verify that a consumer allows only positive, non-zero values for timeout
    if it is supplied.

    :param invalid_timeout: any invalid timeout that should fail validation
    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs
    """
    storage_path = pathlib.Path(test_dir) / "features"
    storage_path.mkdir(parents=True, exist_ok=True)

    mock_storage = {}
    backbone = BackboneFeatureStore(mock_storage)

    channel = FileSystemCommChannel(storage_path / "test-wmgr")

    with pytest.raises(ValueError) as ex:
        # try to create a consumer w/a max recv size of 0
        consumer = EventConsumer(
            channel,
            backbone,
            filters=[OnWriteFeatureStore.FEATURE_STORE_WRITTEN],
        )
        consumer.recv(batch_timeout=invalid_timeout)

    assert "positive" in ex.value.args[0]


@pytest.mark.parametrize(
    "wait_timeout, exp_wait_max",
    [
        # aggregate the 1+1+1 into 3 on remaining parameters
        pytest.param(1, 1 + 1 + 1, id="1s wait, 3 cycle steps"),
        pytest.param(2, 3 + 2, id="2s wait, 4 cycle steps"),
        pytest.param(4, 3 + 2 + 4, id="4s wait, 5 cycle steps"),
        pytest.param(9, 3 + 2 + 4 + 8, id="9s wait, 6 cycle steps"),
        # aggregate an entire cycle into 16
        pytest.param(19.5, 16 + 3 + 2 + 4, id="20s wait, repeat cycle"),
    ],
)
def test_backbone_wait_timeout(wait_timeout: float, exp_wait_max: float) -> None:
    """Verify that attempts to attach to the worker queue from the protoclient
    timeout in an appropriate amount of time. Note: due to the backoff, we verify
    the elapsed time is less than the 15s of a cycle of waits.

    :param wait_timeout: Maximum amount of time (in seconds) to allow the backbone
    to wait for the requested value to exist
    :param exp_wait_max: Maximum amount of time (in seconds) to set as the upper
    bound to allow the delays with backoff to occur
    :param storage_for_dragon_fs: the dragon storage engine to use
    """

    # NOTE: exp_wait_time maps to the cycled backoff of [0.1, 0.2, 0.4, 0.8]
    # with leeway added (by allowing 1s each for the 0.1 and 0.5 steps)
    start_time = time.time()

    storage = {}
    backbone = BackboneFeatureStore(storage)

    with pytest.raises(SmartSimError) as ex:
        backbone.wait_for(["does-not-exist"], wait_timeout)

    assert "timeout" in str(ex.value.args[0]).lower()

    end_time = time.time()
    elapsed = end_time - start_time

    # confirm that we met our timeout
    assert elapsed > wait_timeout, f"below configured timeout {wait_timeout}"

    # confirm that the total wait time is aligned with the sleep cycle
    assert elapsed < exp_wait_max, f"above expected max wait {exp_wait_max}"
