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


import multiprocessing as mp
import random
import time
import typing as t
import unittest.mock as mock
import uuid

import pytest

dragon = pytest.importorskip("dragon")

from smartsim._core.mli.comm.channel.dragon_channel import DragonCommChannel
from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
    EventBroadcaster,
    EventCategory,
    EventConsumer,
    OnCreateConsumer,
    OnWriteFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    time as bbtime,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import dragon_ddict
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


@pytest.fixture(scope="module")
def storage_for_dragon_fs() -> t.Dict[str, str]:
    """Fixture to instantiate a dragon distributed dictionary.

    NOTE: using module scoped fixtures drastically improves test run-time"""
    return dragon_ddict.DDict(1, 2, total_mem=2 * 1024**3)


@pytest.fixture(scope="module")
def the_worker_channel() -> DragonFLIChannel:
    """Fixture to create a valid descriptor for a worker channel
    that can be attached to.

    NOTE: using module scoped fixtures drastically improves test run-time"""
    # wmgr_channel_ = create_local()
    # wmgr_channel = DragonCommChannel(wmgr_channel_)
    # return wmgr_channel
    channel_ = create_local()
    fli_ = fli.FLInterface(main_ch=channel_, manager_ch=None)
    comm_channel = DragonFLIChannel(fli_, True)
    return comm_channel


@pytest.fixture(scope="module")
def the_backbone(
    storage_for_dragon_fs: t.Any, the_worker_channel: DragonFLIChannel
) -> BackboneFeatureStore:
    """Fixture to create a distributed dragon dictionary and wrap it
    in a BackboneFeatureStore.

    NOTE: using module scoped fixtures drastically improves test run-time"""

    backbone = BackboneFeatureStore(storage_for_dragon_fs, allow_reserved_writes=True)
    backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] = the_worker_channel.descriptor

    return backbone


def test_eventconsumer_eventpublisher_integration(
    the_backbone: BackboneFeatureStore, test_dir: str
) -> None:
    """Verify that the publisher and consumer integrate as expected when
    multiple publishers and consumers are sending simultaneously. This
    test closely tracks the test in tests/test_featurestore.py also named
    test_eventconsumer_eventpublisher_integration but requires dragon entities

    :param storage_for_dragon_fs: the dragon storage engine to use
    :param test_dir: pytest fixture automatically generating unique working
    directories for individual test outputs"""

    # verify ability to write and read from ddict
    the_backbone["test_dir"] = test_dir
    assert the_backbone["test_dir"] == test_dir

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
        filters=[EventCategory.FEATURE_STORE_WRITTEN],
    )
    capp_consumer = EventConsumer(
        capp_channel,
        the_backbone,
    )
    back_consumer = EventConsumer(
        back_channel,
        the_backbone,
        filters=[EventCategory.CONSUMER_CREATED],
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
    event_1 = OnCreateConsumer(wmgr_consumer_descriptor, [])
    mock_worker_mgr.send(event_1)

    # simulate the app updating a model a few times
    event_2 = OnWriteFeatureStore(the_backbone.descriptor, "key-1")
    event_3 = OnWriteFeatureStore(the_backbone.descriptor, "key-2")
    event_4 = OnWriteFeatureStore(the_backbone.descriptor, "key-1")

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


def test_backbone_wait_for_no_keys(
    the_backbone: BackboneFeatureStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that asking the backbone to wait for a value succeeds
    immediately and does not cause a wait to occur if the supplied key
    list is empty

    :param storage_for_dragon_fs: the storage engine to use, prepopulated with
    """
    # set a very low timeout to confirm that it does not wait

    with monkeypatch.context() as ctx:
        # all keys should be found and the timeout should never be checked.
        ctx.setattr(bbtime, "sleep", mock.MagicMock())

        values = the_backbone.wait_for([])
        assert len(values) == 0

        # confirm that no wait occurred
        bbtime.sleep.assert_not_called()


def test_backbone_wait_for_prepopulated(
    the_backbone: BackboneFeatureStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that asking the backbone to wait for a value succeed
    immediately and do not cause a wait to occur if the data exists

    :param storage_for_dragon_fs: the storage engine to use, prepopulated with
    """
    # set a very low timeout to confirm that it does not wait

    with monkeypatch.context() as ctx:
        # all keys should be found and the timeout should never be checked.
        ctx.setattr(bbtime, "sleep", mock.MagicMock())

        values = the_backbone.wait_for([BackboneFeatureStore.MLI_WORKER_QUEUE], 0.1)

        # confirm that wait_for with one key returns one value
        assert len(values) == 1

        # confirm that the descriptor is non-null w/some non-trivial value
        assert len(values[BackboneFeatureStore.MLI_WORKER_QUEUE]) > 5

        # confirm that no wait occurred
        bbtime.sleep.assert_not_called()


def test_backbone_wait_for_prepopulated_dupe(
    the_backbone: BackboneFeatureStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that asking the backbone to wait for keys that are duplicated
    results in a single value being returned for each key

    :param storage_for_dragon_fs: the storage engine to use, prepopulated with
    """
    # set a very low timeout to confirm that it does not wait

    key1, key2 = "key-1", "key-2"
    value1, value2 = "i-am-value-1", "i-am-value-2"
    the_backbone[key1] = value1
    the_backbone[key2] = value2

    with monkeypatch.context() as ctx:
        # all keys should be found and the timeout should never be checked.
        ctx.setattr(bbtime, "sleep", mock.MagicMock())

        values = the_backbone.wait_for([key1, key2, key1])  # key1 is duplicated

        # confirm that wait_for with one key returns one value
        assert len(values) == 2
        assert key1 in values
        assert key2 in values

        assert values[key1] == value1
        assert values[key2] == value2


def set_value_after_delay(
    descriptor: str, key: str, value: str, delay: float = 5
) -> None:
    """Helper method to persist a random value into the backbone

    :param descriptor: the backbone feature store descriptor to attach to
    :param key: the key to write to
    :param value: a value to write to the key"""
    time.sleep(delay)

    backbone = BackboneFeatureStore.from_descriptor(descriptor)
    backbone[key] = value
    logger.debug(f"set_value_after_delay wrote `{value} to backbone[`{key}`]")


@pytest.mark.parametrize(
    "delay",
    [
        pytest.param(
            0,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            1,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            2,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            4,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            8,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
    ],
)
def test_backbone_wait_for_partial_prepopulated(
    the_backbone: BackboneFeatureStore, delay: float
) -> None:
    """Verify that when data is not all in the backbone, the `wait_for` operation
    continues to poll until it finds everything it needs

    :param storage_for_dragon_fs: the storage engine to use, prepopulated with
    :param delay: the number of seconds the second process will wait before
    setting the target value in the backbone featurestore
    """
    # set a very low timeout to confirm that it does not wait
    wait_timeout = 10

    key, value = str(uuid.uuid4()), str(random.random() * 10)

    logger.debug(f"Starting process to write {key} after {delay}s")
    p = mp.Process(
        target=set_value_after_delay, args=(the_backbone.descriptor, key, value, delay)
    )
    p.start()

    p2 = mp.Process(
        target=the_backbone.wait_for,
        args=([BackboneFeatureStore.MLI_WORKER_QUEUE, key],),
        kwargs={"timeout": wait_timeout},
    )
    p2.start()

    p.join()
    p2.join()

    # both values should be written at this time
    ret_vals = the_backbone.wait_for(
        [key, BackboneFeatureStore.MLI_WORKER_QUEUE, key], 0.1
    )
    # confirm that wait_for with two keys returns two values
    assert len(ret_vals) == 2, "values should contain values for both awaited keys"

    # confirm the pre-populated value has the correct output
    assert (
        ret_vals[BackboneFeatureStore.MLI_WORKER_QUEUE] == "12345"
    )  # mock descriptor value from fixture

    # confirm the population process completed and the awaited value is correct
    assert ret_vals[key] == value, "verify order of values "


@pytest.mark.parametrize(
    "num_keys",
    [
        pytest.param(
            0,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            1,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            3,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            7,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
        pytest.param(
            11,
            marks=pytest.mark.skip(
                "Must use entrypoint instead of mp.Process to run on build agent"
            ),
        ),
    ],
)
def test_backbone_wait_for_multikey(
    the_backbone: BackboneFeatureStore,
    num_keys: int,
    test_dir: str,
) -> None:
    """Verify that asking the backbone to wait for multiple keys results
    in that number of values being returned

    :param storage_for_dragon_fs: the storage engine to use, prepopulated with
    :param num_keys: the number of extra keys to set & request in the backbone
    """
    # maximum delay allowed for setter processes
    max_delay = 5

    extra_keys = [str(uuid.uuid4()) for _ in range(num_keys)]
    extra_values = [str(uuid.uuid4()) for _ in range(num_keys)]
    extras = dict(zip(extra_keys, extra_values))
    delays = [random.random() * max_delay for _ in range(num_keys)]
    processes = []

    for key, value, delay in zip(extra_keys, extra_values, delays):
        assert delay < max_delay, "write delay exceeds test timeout"
        logger.debug(f"Delaying {key} write by {delay} seconds")
        p = mp.Process(
            target=set_value_after_delay,
            args=(the_backbone.descriptor, key, value, delay),
        )
        p.start()
        processes.append(p)

    p2 = mp.Process(
        target=the_backbone.wait_for,
        args=(extra_keys,),
        kwargs={"timeout": max_delay * 2},
    )
    p2.start()
    for p in processes:
        p.join(timeout=max_delay * 2)
    p2.join(
        timeout=max_delay * 2
    )  # give it 10 seconds longer than p2 timeout for backoff

    # use without a wait to verify all values are written
    num_keys = len(extra_keys)
    actual_values = the_backbone.wait_for(extra_keys, timeout=0.01)
    assert len(extra_keys) == num_keys

    # confirm that wait_for returns all the expected values
    assert len(actual_values) == num_keys

    # confirm that the returned values match (e.g. are returned in the right order)
    for k in extras:
        assert extras[k] == actual_values[k]
