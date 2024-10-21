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

import base64
import pathlib
import uuid

import pytest

from smartsim.error.errors import SmartSimError

dragon = pytest.importorskip("dragon")

# isort: off
import dragon.channels as dch
import dragon.infrastructure.parameters as dp
import dragon.managed_memory as dm
import dragon.fli as fli

# isort: on

from smartsim._core.mli.comm.channel import dragon_util
from smartsim.log import get_logger

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon
logger = get_logger(__name__)


@pytest.fixture(scope="function")
def the_pool() -> dm.MemoryPool:
    """Creates a memory pool."""
    raw_pool_descriptor = dp.this_process.default_pd
    descriptor_ = base64.b64decode(raw_pool_descriptor)

    pool = dm.MemoryPool.attach(descriptor_)
    return pool


@pytest.fixture(scope="function")
def the_channel() -> dch.Channel:
    """Creates a Channel attached to the local memory pool."""
    channel = dch.Channel.make_process_local()
    return channel


@pytest.fixture(scope="function")
def the_fli(the_channel) -> fli.FLInterface:
    """Creates an FLI attached to the local memory pool."""
    fli_ = fli.FLInterface(main_ch=the_channel, manager_ch=None)
    return fli_


def test_descriptor_to_channel_empty() -> None:
    """Verify that `descriptor_to_channel` raises an exception when
    provided with an empty descriptor."""
    descriptor = ""

    with pytest.raises(ValueError) as ex:
        dragon_util.descriptor_to_channel(descriptor)

    assert "empty" in ex.value.args[0]


@pytest.mark.parametrize(
    "descriptor",
    ["a", "ab", "abc", "x1", pathlib.Path(".").absolute().as_posix()],
)
def test_descriptor_to_channel_b64fail(descriptor: str) -> None:
    """Verify that `descriptor_to_channel` raises an exception when
    provided with an incorrectly encoded descriptor.

    :param descriptor: A descriptor that is not properly base64 encoded
    """

    with pytest.raises(ValueError) as ex:
        dragon_util.descriptor_to_channel(descriptor)

    assert "base64" in ex.value.args[0]


@pytest.mark.parametrize(
    "descriptor",
    [str(uuid.uuid4())],
)
def test_descriptor_to_channel_channel_fail(descriptor: str) -> None:
    """Verify that `descriptor_to_channel` raises an exception when a correctly
    formatted descriptor that does not describe a real channel is passed.

    :param descriptor: A descriptor that is not properly base64 encoded
    """

    with pytest.raises(SmartSimError) as ex:
        dragon_util.descriptor_to_channel(descriptor)

    # ensure we're receiving the right exception
    assert "address" in ex.value.args[0]
    assert "channel" in ex.value.args[0]


def test_descriptor_to_channel_channel_not_available(the_channel: dch.Channel) -> None:
    """Verify that `descriptor_to_channel` raises an exception when a channel
    is no longer available.

    :param the_channel: A dragon channel
    """

    # get a good descriptor & wipe out the channel so it can't be attached
    descriptor = dragon_util.channel_to_descriptor(the_channel)
    the_channel.destroy()

    with pytest.raises(SmartSimError) as ex:
        dragon_util.descriptor_to_channel(descriptor)

    assert "address" in ex.value.args[0]


def test_descriptor_to_channel_happy_path(the_channel: dch.Channel) -> None:
    """Verify that `descriptor_to_channel` works as expected when provided
    a valid descriptor

    :param the_channel: A dragon channel
    """

    # get a good descriptor
    descriptor = dragon_util.channel_to_descriptor(the_channel)

    reattached = dragon_util.descriptor_to_channel(descriptor)
    assert reattached

    # and just make sure creation of the descriptor is transitive
    assert dragon_util.channel_to_descriptor(reattached) == descriptor


def test_descriptor_to_fli_empty() -> None:
    """Verify that `descriptor_to_fli` raises an exception when
    provided with an empty descriptor."""
    descriptor = ""

    with pytest.raises(ValueError) as ex:
        dragon_util.descriptor_to_fli(descriptor)

    assert "empty" in ex.value.args[0]


@pytest.mark.parametrize(
    "descriptor",
    ["a", "ab", "abc", "x1", pathlib.Path(".").absolute().as_posix()],
)
def test_descriptor_to_fli_b64fail(descriptor: str) -> None:
    """Verify that `descriptor_to_fli` raises an exception when
    provided with an incorrectly encoded descriptor.

    :param descriptor: A descriptor that is not properly base64 encoded
    """

    with pytest.raises(ValueError) as ex:
        dragon_util.descriptor_to_fli(descriptor)

    assert "base64" in ex.value.args[0]


@pytest.mark.parametrize(
    "descriptor",
    [str(uuid.uuid4())],
)
def test_descriptor_to_fli_fli_fail(descriptor: str) -> None:
    """Verify that `descriptor_to_fli` raises an exception when a correctly
    formatted descriptor that does not describe a real FLI is passed.

    :param descriptor: A descriptor that is not properly base64 encoded
    """

    with pytest.raises(SmartSimError) as ex:
        dragon_util.descriptor_to_fli(descriptor)

    # ensure we're receiving the right exception
    assert "address" in ex.value.args[0]
    assert "fli" in ex.value.args[0].lower()


def test_descriptor_to_fli_fli_not_available(
    the_fli: fli.FLInterface, the_channel: dch.Channel
) -> None:
    """Verify that `descriptor_to_fli` raises an exception when a channel
    is no longer available.

    :param the_fli: A dragon FLInterface
    :param the_channel: A dragon channel
    """

    # get a good descriptor & wipe out the FLI so it can't be attached
    descriptor = dragon_util.channel_to_descriptor(the_fli)
    the_fli.destroy()
    the_channel.destroy()

    with pytest.raises(SmartSimError) as ex:
        dragon_util.descriptor_to_fli(descriptor)

    # ensure we're receiving the right exception
    assert "address" in ex.value.args[0]


def test_descriptor_to_fli_happy_path(the_fli: dch.Channel) -> None:
    """Verify that `descriptor_to_fli` works as expected when provided
    a valid descriptor

    :param the_fli: A dragon FLInterface
    """

    # get a good descriptor
    descriptor = dragon_util.channel_to_descriptor(the_fli)

    reattached = dragon_util.descriptor_to_fli(descriptor)
    assert reattached

    # and just make sure creation of the descriptor is transitive
    assert dragon_util.channel_to_descriptor(reattached) == descriptor


def test_pool_to_descriptor_empty() -> None:
    """Verify that `pool_to_descriptor` raises an exception when
    provided with a null pool."""

    with pytest.raises(ValueError) as ex:
        dragon_util.pool_to_descriptor(None)


def test_pool_to_happy_path(the_pool) -> None:
    """Verify that `pool_to_descriptor` creates a descriptor
    when supplied with a valid memory pool."""

    descriptor = dragon_util.pool_to_descriptor(the_pool)
    assert descriptor
