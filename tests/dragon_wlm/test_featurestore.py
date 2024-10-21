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

from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    time as bbtime,
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


def test_backbone_wait_for_no_keys(
    the_backbone: BackboneFeatureStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that asking the backbone to wait for a value succeeds
    immediately and does not cause a wait to occur if the supplied key
    list is empty.

    :param the_backbone: the storage engine to use, prepopulated with
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
    immediately and do not cause a wait to occur if the data exists.

    :param the_backbone: the storage engine to use, prepopulated with
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
    results in a single value being returned for each key.

    :param the_backbone: the storage engine to use, prepopulated with
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
    :param value: a value to write to the key
    :param delay: amount of delay to apply before writing the key
    """
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
    continues to poll until it finds everything it needs.

    :param the_backbone: the storage engine to use, prepopulated with
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
    in that number of values being returned.

    :param the_backbone: the storage engine to use, prepopulated with
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
