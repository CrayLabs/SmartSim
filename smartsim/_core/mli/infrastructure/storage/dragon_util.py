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

# pylint: disable=import-error
# isort: off
import dragon.data.ddict.ddict as dragon_ddict

# isort: on

from smartsim.log import get_logger

logger = get_logger(__name__)


def ddict_to_descriptor(ddict: dragon_ddict.DDict) -> str:
    """Convert a DDict to a descriptor string.

    :param ddict: The dragon dictionary to convert
    :returns: The descriptor string
    :raises ValueError: If a ddict is not provided
    """
    if ddict is None:
        raise ValueError("DDict is not available to create a descriptor")

    # unlike other dragon objects, the dictionary serializes to a string
    # instead of bytes
    return str(ddict.serialize())


def descriptor_to_ddict(descriptor: str) -> dragon_ddict.DDict:
    """Create and attach a new DDict instance given
    the string-encoded descriptor.

    :param descriptor: The descriptor of a dictionary to attach to
    :returns: The attached dragon dictionary"""
    return dragon_ddict.DDict.attach(descriptor)


def create_ddict(
    num_nodes: int, mgr_per_node: int, mem_per_node: int
) -> dragon_ddict.DDict:
    """Create a distributed dragon dictionary.

    :param num_nodes: The number of distributed nodes to distribute the dictionary to.
     At least one node is required.
    :param mgr_per_node: The number of manager processes per node
    :param mem_per_node: The amount of memory (in bytes) to allocate per node. Total
     memory available will be calculated as `num_nodes * node_mem`

    :returns: The instantiated dragon dictionary
    :raises ValueError: If invalid num_nodes is supplied
    :raises ValueError: If invalid mem_per_node is supplied
    :raises ValueError: If invalid mgr_per_node is supplied
    """
    if num_nodes < 1:
        raise ValueError("A dragon dictionary must have at least 1 node")

    if mgr_per_node < 1:
        raise ValueError("A dragon dict requires at least 2 managers per ndode")

    if mem_per_node < dragon_ddict.DDICT_MIN_SIZE:
        raise ValueError(
            "A dragon dictionary requires at least "
            f"{dragon_ddict.DDICT_MIN_SIZE / (1024**2)} MB"
        )

    mem_total = num_nodes * mem_per_node

    logger.debug(
        f"Creating dragon dictionary with {num_nodes} nodes, {mem_total} bytes memory"
    )

    distributed_dict = dragon_ddict.DDict(num_nodes, mgr_per_node, total_mem=mem_total)
    logger.debug(
        "Successfully created dragon dictionary with "
        f"{num_nodes} nodes, {mem_total} bytes total memory"
    )
    return distributed_dict
