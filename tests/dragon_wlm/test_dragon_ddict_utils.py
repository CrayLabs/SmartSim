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

import pytest

dragon = pytest.importorskip("dragon")

# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim._core.mli.infrastructure.storage import dragon_util
from smartsim.log import get_logger

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon
logger = get_logger(__name__)


@pytest.mark.parametrize(
    "num_nodes, num_managers, mem_per_node",
    [
        pytest.param(1, 1, 3 * 1024**2, id="3MB, Bare minimum allocation"),
        pytest.param(2, 2, 128 * 1024**2, id="128 MB allocation, 2 nodes, 2 mgr"),
        pytest.param(2, 1, 512 * 1024**2, id="512 MB allocation, 2 nodes, 1 mgr"),
    ],
)
def test_dragon_storage_util_create_ddict(
    num_nodes: int,
    num_managers: int,
    mem_per_node: int,
):
    """Verify that a dragon dictionary is successfully created.

    :param num_nodes: Number of ddict nodes to attempt to create
    :param num_managers: Number of managers per node to request
    :param num_managers: Memory to allocate per node
    """
    ddict = dragon_util.create_ddict(num_nodes, num_managers, mem_per_node)

    assert ddict is not None


@pytest.mark.parametrize(
    "num_nodes, num_managers, mem_per_node",
    [
        pytest.param(-1, 1, 3 * 1024**2, id="Negative Node Count"),
        pytest.param(0, 1, 3 * 1024**2, id="Invalid Node Count"),
        pytest.param(1, -1, 3 * 1024**2, id="Negative Mgr Count"),
        pytest.param(1, 0, 3 * 1024**2, id="Invalid Mgr Count"),
        pytest.param(1, 1, -3 * 1024**2, id="Negative Mem Per Node"),
        pytest.param(1, 1, (3 * 1024**2) - 1, id="Invalid Mem Per Node"),
        pytest.param(1, 1, 0 * 1024**2, id="No Mem Per Node"),
    ],
)
def test_dragon_storage_util_create_ddict_validators(
    num_nodes: int,
    num_managers: int,
    mem_per_node: int,
):
    """Verify that a dragon dictionary is successfully created.

    :param num_nodes: Number of ddict nodes to attempt to create
    :param num_managers: Number of managers per node to request
    :param num_managers: Memory to allocate per node
    """
    with pytest.raises(ValueError):
        dragon_util.create_ddict(num_nodes, num_managers, mem_per_node)


def test_dragon_storage_util_get_ddict_descriptor(the_storage: dragon_ddict.DDict):
    """Verify that a descriptor is created.

    :param the_storage: A pre-allocated ddict
    """
    value = dragon_util.ddict_to_descriptor(the_storage)

    assert isinstance(value, str)
    assert len(value) > 0


def test_dragon_storage_util_get_ddict_from_descriptor(the_storage: dragon_ddict.DDict):
    """Verify that a ddict is created from a descriptor.

    :param the_storage: A pre-allocated ddict
    """
    descriptor = dragon_util.ddict_to_descriptor(the_storage)

    value = dragon_util.descriptor_to_ddict(descriptor)

    assert value is not None
    assert isinstance(value, dragon_ddict.DDict)
    assert dragon_util.ddict_to_descriptor(value) == descriptor
