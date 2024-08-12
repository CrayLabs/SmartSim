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
import random
import threading
import typing as t

import pytest

from smartsim._core.launcher.dragon.pqueue import NodePrioritizer, PrioritizerFilter
from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


logger = get_logger(__name__)


class MockNode:
    def __init__(self, hostname: str, num_cpus: int, num_gpus: int) -> None:
        self.hostname = hostname
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus


def mock_node_hosts(
    num_cpu_nodes: int, num_gpu_nodes: int
) -> t.Tuple[t.List[MockNode], t.List[MockNode]]:
    cpu_hosts = [f"cpu-node-{i}" for i in range(num_cpu_nodes)]
    gpu_hosts = [f"gpu-node-{i}" for i in range(num_gpu_nodes)]

    return cpu_hosts, gpu_hosts


def mock_node_builder(num_cpu_nodes: int, num_gpu_nodes: int) -> t.List[MockNode]:
    nodes = []
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)

    nodes.extend(MockNode(hostname, 4, 0) for hostname in cpu_hosts)
    nodes.extend(MockNode(hostname, 4, 4) for hostname in gpu_hosts)

    return nodes


def test_node_prioritizer_init_null() -> None:
    """Verify that the priorizer reports failures to send a valid node set
    if a null value is passed"""
    lock = threading.RLock()
    with pytest.raises(SmartSimError) as ex:
        NodePrioritizer(None, lock)

    assert "Missing" in ex.value.args[0]


def test_node_prioritizer_init_empty() -> None:
    """Verify that the priorizer reports failures to send a valid node set
    if an empty list is passed"""
    lock = threading.RLock()
    with pytest.raises(SmartSimError) as ex:
        NodePrioritizer([], lock)

    assert "Missing" in ex.value.args[0]


@pytest.mark.parametrize(
    "num_cpu_nodes,num_gpu_nodes", [(1, 1), (2, 1), (1, 2), (8, 4), (1000, 200)]
)
def test_node_prioritizer_init_ok(num_cpu_nodes: int, num_gpu_nodes: int) -> None:
    """Verify that initialization with a valid node list results in the
    appropriate cpu & gpu ref counts, and complete ref map"""
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    # perform prioritizer initialization
    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # get a copy of all the expected host names
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    all_hosts = cpu_hosts + gpu_hosts
    assert len(all_hosts) == num_cpu_nodes + num_gpu_nodes

    # verify tracking data is initialized correctly for all nodes
    for hostname in all_hosts:
        # show that the ref map is tracking the node
        assert hostname in p._nodes

        tracking_info = p.get_tracking_info(hostname)

        # show that the node is created w/zero ref counts
        assert tracking_info.num_refs == 0

        # show that the node is created and marked as not dirty (unchanged)
        # assert tracking_info.is_dirty == False

    # iterate through known cpu node keys and verify prioritizer initialization
    for hostname in cpu_hosts:
        # show that the device ref counters are appropriately assigned
        cpu_ref = next((n for n in p._cpu_refs if n.hostname == hostname), None)
        assert cpu_ref, "CPU-only node not found in cpu ref set"

        gpu_ref = next((n for n in p._gpu_refs if n.hostname == hostname), None)
        assert not gpu_ref, "CPU-only node should not be found in gpu ref set"

    # iterate through known GPU node keys and verify prioritizer initialization
    for hostname in gpu_hosts:
        # show that the device ref counters are appropriately assigned
        gpu_ref = next((n for n in p._gpu_refs if n.hostname == hostname), None)
        assert gpu_ref, "GPU-only node not found in gpu ref set"

        cpu_ref = next((n for n in p._cpu_refs if n.hostname == hostname), None)
        assert not cpu_ref, "GPU-only node should not be found in cpu ref set"

    # verify we have all hosts in the ref map
    assert set(p._nodes.keys()) == set(all_hosts)

    # verify we have no extra hosts in ref map
    assert len(p._nodes.keys()) == len(set(all_hosts))


def test_node_prioritizer_direct_increment() -> None:
    """Verify that performing the increment operation causes the expected
    side effect on the intended records"""

    num_cpu_nodes, num_gpu_nodes = 32, 8
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    exclude_index = 2
    exclude_host0 = cpu_hosts[exclude_index]
    exclude_host1 = gpu_hosts[exclude_index]
    exclusions = [exclude_host0, exclude_host1]

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # let's increment each element in a predictable way and verify
    for node in nodes:
        if node.hostname in exclusions:
            # expect 1 cpu and 1 gpu node at zero and not incremented
            continue

        if node.num_gpus == 0:
            num_increments = random.randint(0, num_cpu_nodes - 1)
        else:
            num_increments = random.randint(0, num_gpu_nodes - 1)

        # increment this node some random number of times
        for _ in range(num_increments):
            p.increment(node.hostname)

        # ... and verify the correct incrementing is applied
        tracking_info = p.get_tracking_info(node.hostname)
        assert tracking_info.num_refs == num_increments

    # verify the excluded cpu node was never changed
    tracking_info0 = p.get_tracking_info(exclude_host0)
    assert tracking_info0.num_refs == 0

    # verify the excluded gpu node was never changed
    tracking_info1 = p.get_tracking_info(exclude_host1)
    assert tracking_info1.num_refs == 0


def test_node_prioritizer_indirect_increment() -> None:
    """Verify that performing the increment operation indirectly affects
    each available node until we run out of nodes to return"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # verify starting state
    for node in p._nodes.values():
        tracking_info = p.get_tracking_info(node.hostname)

        assert node.num_refs == 0  # <--- ref count starts at zero
        assert tracking_info.num_refs == 0  # <--- ref count starts at zero

    # perform indirect
    for node in p._nodes.values():
        tracking_info = p.get_tracking_info(node.hostname)

        # apply `next` operation and verify tracking info reflects new ref
        node = p.next(PrioritizerFilter.CPU)
        tracking_info = p.get_tracking_info(node.hostname)

        # verify side-effects
        assert tracking_info.num_refs > 0  # <--- ref count should now be > 0

        # we expect it to give back only "clean" nodes from next*
        assert tracking_info.is_dirty == False  # NOTE: this is "hidden" by protocol

    # every node should be incremented now. prioritizer shouldn't have anything to give
    tracking_info = p.next(PrioritizerFilter.CPU)
    assert tracking_info is None  # <--- get_next shouldn't have any nodes to give


def test_node_prioritizer_indirect_decrement_availability() -> None:
    """Verify that a node who is decremented (dirty) is made assignable
    on a subsequent request"""

    num_cpu_nodes, num_gpu_nodes = 1, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # increment our only node...
    p.increment(cpu_hosts[0])

    tracking_info = p.next()
    assert tracking_info is None, "No nodes should be assignable"

    # perform a decrement...
    p.decrement(cpu_hosts[0])

    # ... and confirm that the node is available again
    tracking_info = p.next()
    assert tracking_info is not None, "A node should be assignable"


def test_node_prioritizer_multi_increment() -> None:
    """Verify that retrieving multiple nodes via `next_n` API correctly
    increments reference counts and returns appropriate results"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    assert p.get_tracking_info(cpu_hosts[0]).num_refs > 0

    p.increment(cpu_hosts[2])
    assert p.get_tracking_info(cpu_hosts[2]).num_refs > 0

    p.increment(cpu_hosts[4])
    assert p.get_tracking_info(cpu_hosts[4]).num_refs > 0

    # use next_n w/the minimum allowed value
    all_tracking_info = p.next_n(1, PrioritizerFilter.CPU)  # <---- next_n(1)

    # confirm the number requested is honored
    assert len(all_tracking_info) == 1
    # ensure no unavailable node is returned
    assert all_tracking_info[0].hostname not in [
        cpu_hosts[0],
        cpu_hosts[2],
        cpu_hosts[4],
    ]

    # use next_n w/value that exceeds available number of open nodes
    # 3 direct increments in setup, 1 out of next_n(1), 4 left
    all_tracking_info = p.next_n(5, PrioritizerFilter.CPU)

    # confirm that no nodes are returned, even though 4 out of 5 requested are available
    assert len(all_tracking_info) == 0


def test_node_prioritizer_multi_increment_validate_n() -> None:
    """Verify that retrieving multiple nodes via `next_n` API correctly
    reports failures when the request size is above pool size"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # we have 8 total cpu nodes available... request too many nodes
    all_tracking_info = p.next_n(9, PrioritizerFilter.CPU)
    assert len(all_tracking_info) == 0

    all_tracking_info = p.next_n(num_cpu_nodes * 1000, PrioritizerFilter.CPU)
    assert len(all_tracking_info) == 0


def test_node_prioritizer_indirect_direct_interleaved_increments() -> None:
    """Verify that interleaving indirect and direct increments results in
    expected ref counts"""

    num_cpu_nodes, num_gpu_nodes = 8, 4
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # perform some set of non-popped increments
    p.increment(gpu_hosts[1])
    p.increment(gpu_hosts[3])
    p.increment(gpu_hosts[3])

    # increment 0th item 1x
    p.increment(cpu_hosts[0])

    # increment 3th item 2x
    p.increment(cpu_hosts[3])
    p.increment(cpu_hosts[3])

    # increment last item 3x
    p.increment(cpu_hosts[7])
    p.increment(cpu_hosts[7])
    p.increment(cpu_hosts[7])

    tracking_info = p.get_tracking_info(gpu_hosts[1])
    assert tracking_info.num_refs == 1

    tracking_info = p.get_tracking_info(gpu_hosts[3])
    assert tracking_info.num_refs == 2

    nodes = [n for n in p._nodes.values() if n.num_refs == 0 and n.num_gpus == 0]

    # we should skip the 0-th item in the heap due to direct increment
    tracking_info = p.next(PrioritizerFilter.CPU)
    assert tracking_info.num_refs == 1
    # confirm we get a cpu node
    assert "cpu-node" in tracking_info.hostname

    # this should pull the next item right out
    tracking_info = p.next(PrioritizerFilter.CPU)
    assert tracking_info.num_refs == 1
    assert "cpu-node" in tracking_info.hostname

    # ensure we pull from gpu nodes and the 0th item is returned
    tracking_info = p.next(PrioritizerFilter.GPU)
    assert tracking_info.num_refs == 1
    assert "gpu-node" in tracking_info.hostname

    # we should step over the 3-th node on this iteration
    tracking_info = p.next(PrioritizerFilter.CPU)
    assert tracking_info.num_refs == 1
    assert "cpu-node" in tracking_info.hostname

    # and ensure that heap also steps over a direct increment
    tracking_info = p.next(PrioritizerFilter.GPU)
    assert tracking_info.num_refs == 1
    assert "gpu-node" in tracking_info.hostname

    # and another GPU request should return nothing
    tracking_info = p.next(PrioritizerFilter.GPU)
    assert tracking_info is None


def test_node_prioritizer_decrement_floor() -> None:
    """Verify that repeatedly decrementing ref counts does not
    allow negative ref counts"""

    num_cpu_nodes, num_gpu_nodes = 8, 4
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # try a ton of decrements on all the items in the prioritizer
    for _ in range(len(nodes) * 100):
        index = random.randint(0, num_cpu_nodes - 1)
        p.decrement(cpu_hosts[index])

        index = random.randint(0, num_gpu_nodes - 1)
        p.decrement(gpu_hosts[index])

    for node in nodes:
        tracking_info = p.get_tracking_info(node.hostname)
        assert tracking_info.num_refs == 0


@pytest.mark.parametrize("num_requested", [1, 2, 3])
def test_node_prioritizer_multi_increment_subheap(num_requested: int) -> None:
    """Verify that retrieving multiple nodes via `next_n` API correctly
    increments reference counts and returns appropriate results
    when requesting an in-bounds number of nodes"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    p.increment(cpu_hosts[2])
    p.increment(cpu_hosts[4])

    hostnames = [cpu_hosts[0], cpu_hosts[1], cpu_hosts[2], cpu_hosts[3], cpu_hosts[5]]

    # request n == {num_requested} nodes from set of 3 available
    all_tracking_info = p.next_n(
        num_requested,
        hosts=hostnames,
    )  # <---- w/0,2,4 assigned, only 1,3,5 from hostnames can work

    # all parameterizations should result in a matching output size
    assert len(all_tracking_info) == num_requested


def test_node_prioritizer_multi_increment_subheap_assigned() -> None:
    """Verify that retrieving multiple nodes via `next_n` API does
    not return anything when the number requested cannot be satisfied
    by the given subheap due to prior assignment"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    p.increment(cpu_hosts[2])

    hostnames = [
        cpu_hosts[0],
        "x" + cpu_hosts[2],
    ]  # <--- we can't get 2 from 1 valid node name

    # request n == {num_requested} nodes from set of 3 available
    num_requested = 2
    all_tracking_info = p.next_n(num_requested, hosts=hostnames)

    # w/0,2 assigned, nothing can be returned
    assert len(all_tracking_info) == 0


def test_node_prioritizer_empty_subheap_next_w_hosts() -> None:
    """Verify that retrieving multiple nodes via `next_n` API does
    not allow an empty host list"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    p.increment(cpu_hosts[2])

    hostnames = []

    # request n == {num_requested} nodes from set of 3 available
    num_requested = 1
    with pytest.raises(ValueError) as ex:
        p.next(hosts=hostnames)

    assert "No host names provided" == ex.value.args[0]


def test_node_prioritizer_empty_subheap_next_n_w_hosts() -> None:
    """Verify that retrieving multiple nodes via `next_n` API does
    not allow an empty host list"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    p.increment(cpu_hosts[2])

    hostnames = []

    # request n == {num_requested} nodes from set of 3 available
    num_requested = 1
    with pytest.raises(ValueError) as ex:
        p.next_n(num_requested, hosts=hostnames)

    assert "No host names provided" == ex.value.args[0]


@pytest.mark.parametrize("num_requested", [-100, -1, 0])
def test_node_prioritizer_empty_subheap_next_n(num_requested: int) -> None:
    """Verify that retrieving a node via `next_n` API does
    not allow a request with num_items < 1"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    p.increment(cpu_hosts[2])

    # request n == {num_requested} nodes from set of 3 available
    with pytest.raises(ValueError) as ex:
        p.next_n(num_requested)

    assert "Number of items requested" in ex.value.args[0]


@pytest.mark.parametrize("num_requested", [-100, -1, 0])
def test_node_prioritizer_empty_subheap_next_n(num_requested: int) -> None:
    """Verify that retrieving multiple nodes via `next_n` API does
    not allow a request with num_items < 1"""

    num_cpu_nodes, num_gpu_nodes = 8, 0
    cpu_hosts, gpu_hosts = mock_node_hosts(num_cpu_nodes, num_gpu_nodes)
    nodes = mock_node_builder(num_cpu_nodes, num_gpu_nodes)

    lock = threading.RLock()
    p = NodePrioritizer(nodes, lock)

    # Mark some nodes as dirty to verify retrieval
    p.increment(cpu_hosts[0])
    p.increment(cpu_hosts[2])

    hostnames = [cpu_hosts[0], cpu_hosts[2]]

    # request n == {num_requested} nodes from set of 3 available
    with pytest.raises(ValueError) as ex:
        p.next_n(num_requested, hosts=hostnames)

    assert "Number of items requested" in ex.value.args[0]
