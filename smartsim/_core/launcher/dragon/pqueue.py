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
import enum
import heapq
import threading
import typing as t

from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


# tracking structure for [num_refs, node_name, is_dirty]
_NodeRefCount = t.List[t.Union[int, str]]


class PrioritizerFilter(str, enum.Enum):
    """A filter used to select a subset of nodes to be queried"""

    CPU = enum.auto()
    GPU = enum.auto()


class Node(t.Protocol):
    """Minimum Node API required to support the NodePrioritizer"""

    @property
    def num_cpus(self) -> int: ...

    @property
    def num_gpus(self) -> int: ...

    @property
    def hostname(self) -> str: ...


class NodePrioritizer:
    def __init__(self, nodes: t.List[Node], lock: threading.RLock) -> None:
        """Initialize the prioritizer

        :param nodes: node attribute information for initializing the priorizer
        :param lock: a lock used to ensure threadsafe operations
        """
        if not nodes:
            raise SmartSimError("Missing nodes to prioritize")

        self._lock = lock
        """Lock used to ensure thread safe changes of the reference counters"""

        self._ref_map: t.Dict[str, _NodeRefCount] = {}
        """Map node names to a ref counter for direct access"""

        self._cpu_refs: t.List[_NodeRefCount] = []
        """Track reference counts to CPU-only nodes"""

        self._gpu_refs: t.List[_NodeRefCount] = []
        """Track reference counts to GPU nodes"""

        self._initialize_reference_counters(nodes)

    def _initialize_reference_counters(self, nodes: t.List[Node]) -> None:
        """Perform initialization of reference counters for nodes in the allocation

        :param nodes: node attribute information for initializing the priorizer"""
        for node in nodes:
            # initialize all node counts to 0 and mark the entries "is_dirty=False"
            tracking_info: _NodeRefCount = [
                0,
                node.hostname,
                0,
            ]  # use list for mutability

            self._ref_map[node.hostname] = tracking_info

            if node.num_gpus:
                self._gpu_refs.append(tracking_info)
            else:
                self._cpu_refs.append(tracking_info)

    def increment(self, host: str) -> None:
        """Directly increment the reference count of a given node and ensure the
        ref counter is marked as dirty to trigger a reordering on retrieval

        :param host: a hostname that should have a reference counter selected"""
        with self._lock:
            tracking_info = self._ref_map[host]
            ref_count, *_ = tracking_info
            tracking_info[0] = int(ref_count) + 1
            tracking_info[2] = 1

    def _all_refs(self) -> t.List[_NodeRefCount]:
        """Combine the CPU and GPU nodes into a single heap

        :returns: list of all reference counters"""
        refs = [*self._cpu_refs, *self._gpu_refs]
        heapq.heapify(refs)
        return refs

    def get_tracking_info(self, host: str) -> _NodeRefCount:
        """Returns the reference counter information for a single node

        :param host: a hostname that should have a reference counter selected
        :returns: a reference counter"""
        return self._ref_map[host]

    def decrement(self, host: str) -> None:
        """Directly increment the reference count of a given node and ensure the
        ref counter is marked as dirty to trigger a reordering

        :param host: a hostname that should have a reference counter decremented"""
        with self._lock:
            tracking_info = self._ref_map[host]
            tracking_info[0] = max(int(tracking_info[0]) - 1, 0)
            tracking_info[2] = 1

    def _create_sub_heap(self, hosts: t.List[str]) -> t.List[_NodeRefCount]:
        """Create a new heap from the primary heap with user-specified nodes

        :param hosts: a list of hostnames used to filter the available nodes
        :returns: a list of assigned reference counters
        """
        nodes_tracking_info: t.List[_NodeRefCount] = []

        # Collect all the tracking info for the requested nodes...
        for host in hosts:
            tracking_info = self._ref_map[host]
            nodes_tracking_info.append(tracking_info)

        # ... and use it to create a new heap from a specified subset of nodes
        heapq.heapify(nodes_tracking_info)

        return nodes_tracking_info

    def next_from(self, hosts: t.List[str]) -> t.Optional[_NodeRefCount]:
        """Return the next node available given a set of desired hosts

        :param hosts: a list of hostnames used to filter the available nodes
        :returns: a list of assigned reference counters
        :raises ValueError: if no host names are provided"""
        if not hosts or len(hosts) == 0:
            raise ValueError("No host names provided")

        sub_heap = self._create_sub_heap(hosts)
        return self._get_next_available_node(sub_heap)

    def next_n_from(self, num_items: int, hosts: t.List[str]) -> t.List[_NodeRefCount]:
        """Return the next N available nodes given a set of desired hosts

        :param num_items: the desird number of nodes to allocate
        :param hosts: a list of hostnames used to filter the available nodes
        :returns: a list of reference counts
        :raises ValueError: if no host names are provided"""
        if not hosts or len(hosts) == 0:
            raise ValueError("No host names provided")

        if num_items < 1:
            raise ValueError(f"Number of items requested {num_items} is invalid")

        sub_heap = self._create_sub_heap(hosts)
        return self._get_next_n_available_nodes(num_items, sub_heap)

    def unassigned(
        self, heap: t.Optional[t.List[_NodeRefCount]] = None
    ) -> t.List[_NodeRefCount]:
        """Select nodes that are currently not assigned a task

        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counts for all unassigned nodes"""
        if heap is None:
            return [node for node in self._ref_map.values() if node[0] == 0]

        return [node for node in heap if node[0] == 0]

    def assigned(
        self, heap: t.Optional[t.List[_NodeRefCount]] = None
    ) -> t.List[_NodeRefCount]:
        """Helper method to identify the nodes that are currently assigned

        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counts for all assigned nodes"""
        if heap is None:
            return [node for node in self._ref_map.values() if node[0] == 1]

        return [node for node in heap if node[0] == 1]

    def _check_satisfiable_n(
        self, num_items: int, heap: t.Optional[t.List[_NodeRefCount]] = None
    ) -> bool:
        """Validates that a request for some number of nodes `n` can be
        satisfied by the prioritizer given the set of nodes available

        :param num_items: the desird number of nodes to allocate
        :param heap: (optional) a subset of the node heap to consider"""
        num_nodes = len(self._ref_map.keys())

        if num_items < 1:
            msg = "Cannot handle request; request requires a positive integer"
            logger.warning(msg)
            return False

        if num_nodes < num_items:
            msg = f"Cannot satisfy request for {num_items} nodes; {num_nodes} in pool"
            logger.warning(msg)
            return False

        num_open = len(self.unassigned(heap))
        if num_open < num_items:
            msg = f"Cannot satisfy request for {num_items} nodes; {num_open} available"
            logger.warning(msg)
            return False

        return True

    def _get_next_available_node(
        self, heap: t.List[_NodeRefCount]
    ) -> t.Optional[_NodeRefCount]:
        """Finds the next node w/the least amount of running processes and
        ensures that any elements that were directly updated are updated in
        the priority structure before being made available

        :param heap: (optional) a subset of the node heap to consider
        :returns: a reference counter for an available node if an unassigned node
        exists, `None` otherwise"""
        tracking_info: t.Optional[_NodeRefCount] = None

        with self._lock:
            tracking_info = heapq.heappop(heap)
            is_dirty = tracking_info[2]

            while is_dirty:
                if is_dirty:
                    # mark dirty items clean and place back into heap to be sorted
                    tracking_info[2] = 0
                    heapq.heappush(heap, tracking_info)

                tracking_info = heapq.heappop(heap)
                is_dirty = tracking_info[2]

            original_ref_count = int(tracking_info[0])
            if original_ref_count == 0:
                # increment the ref count before putting back onto heap
                tracking_info[0] = original_ref_count + 1

            heapq.heappush(heap, tracking_info)

            # next available must enforce only "open" return nodes
            if original_ref_count > 0:
                return None

        return tracking_info

    def _get_next_n_available_nodes(
        self,
        num_items: int,
        heap: t.List[_NodeRefCount],
    ) -> t.List[_NodeRefCount]:
        """Find the next N available nodes w/least amount of references using
        the supplied filter to target a specific node capability

        :param n: number of nodes to reserve
        :returns: a list of reference counters for a available nodes if enough
        unassigned nodes exists, `None` otherwise
        :raises ValueError: if the number of requetsed nodes is not a positive integer
        """
        next_nodes: t.List[_NodeRefCount] = []

        if num_items < 1:
            raise ValueError(f"Number of items requested {num_items} is invalid")

        if not self._check_satisfiable_n(num_items, heap):
            return next_nodes

        while len(next_nodes) < num_items:
            next_node = self._get_next_available_node(heap)
            if next_node:
                next_nodes.append(next_node)
            else:
                break

        return next_nodes

    def next(
        self, filter_on: t.Optional[PrioritizerFilter] = None
    ) -> t.Optional[_NodeRefCount]:
        """Find the next node available w/least amount of references using
        the supplied filter to target a specific node capability

        :param filter_on: the subset of nodes to query for available nodes
        :returns: a reference counter for an available node if an unassigned node
        exists, `None` otherwise"""
        if filter_on == PrioritizerFilter.GPU:
            heap = self._gpu_refs
        elif filter_on == PrioritizerFilter.CPU:
            heap = self._cpu_refs
        else:
            heap = self._all_refs()

        if node := self._get_next_available_node(heap):
            return node

        return None

    def next_n(
        self, num_items: int = 1, filter_on: t.Optional[PrioritizerFilter] = None
    ) -> t.List[_NodeRefCount]:
        """Find the next N available nodes w/least amount of references using
        the supplied filter to target a specific node capability

        :param num_items: number of nodes to reserve
        :param filter_on: the subset of nodes to query for available nodes
        :returns: Collection of reserved nodes"""
        heap = self._cpu_refs
        if filter_on == PrioritizerFilter.GPU:
            heap = self._gpu_refs
        elif filter_on == PrioritizerFilter.CPU:
            heap = self._cpu_refs
        else:
            heap = self._all_refs()

        return self._get_next_n_available_nodes(num_items, heap)
