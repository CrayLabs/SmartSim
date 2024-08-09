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
# import collections
import enum
import heapq
import threading
import typing as t
from dataclasses import dataclass, field

from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


@dataclass
class _TrackedNode:
    """Minimum Node API required to support the NodePrioritizer"""

    num_cpus: int
    """The number of CPUs available on this node"""
    num_gpus: int
    """The number of GPUs available on this node"""
    hostname: str
    """The hostname of this node"""
    num_refs: int = 0
    """The number of processes currently using this node"""

    tracking: t.Set[str] = field(default_factory=set)
    """The unique identifiers of processes using this node"""
    allocated_cpus: t.Set[int] = field(default_factory=set)
    """The CPU indices allocated on this node"""
    allocated_gpus: t.Set[int] = field(default_factory=set)
    """The GPU indices allocated on this node"""
    dirty: bool = False
    """Flag indicating that the node has been updated"""

    def add(
        self,
        tracking_id: t.Optional[str] = None,
        cpus: t.Optional[t.Sequence[int]] = None,
        gpus: t.Optional[t.Sequence[int]] = None,
    ) -> None:
        if tracking_id in self.tracking:
            raise ValueError("Attempted adding task more than once")

        self.num_refs = self.num_refs + 1
        if tracking_id:
            self.tracking = self.tracking.union({tracking_id})
        if cpus:
            self.allocated_cpus.update(set(cpus))
        if gpus:
            self.allocated_gpus.update(set(gpus))
        self.dirty = True

    def remove(
        self,
        tracking_id: t.Optional[str] = None,
        cpus: t.Optional[t.Sequence[int]] = None,
        gpus: t.Optional[t.Sequence[int]] = None,
    ) -> None:
        if tracking_id and tracking_id not in self.tracking:
            raise ValueError("Attempted removal of untracked item")

        self.num_refs = max(self.num_refs - 1, 0)
        self.tracking = self.tracking - {tracking_id}
        if cpus:
            self.allocated_cpus.difference_update(set(cpus))
        if gpus:
            self.allocated_gpus.difference_update(set(gpus))
        self.dirty = True

    def __lt__(self, other: "_TrackedNode") -> bool:
        if self.num_refs < other.num_refs:
            return True

        return False


class PrioritizerFilter(str, enum.Enum):
    """A filter used to select a subset of nodes to be queried"""

    CPU = enum.auto()
    GPU = enum.auto()


class Node(t.Protocol):
    """Minimum Node API required to support the NodePrioritizer"""

    hostname: str
    num_cpus: int
    num_gpus: int


class NodeReferenceCount(t.Protocol):
    """Contains details pertaining to references to a node"""

    hostname: str
    num_refs: int


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

        self._cpu_refs: t.List[_TrackedNode] = []
        """Track reference counts to CPU-only nodes"""

        self._gpu_refs: t.List[_TrackedNode] = []
        """Track reference counts to GPU nodes"""

        self._nodes: t.Dict[str, _TrackedNode] = {}

        self._initialize_reference_counters(nodes)

    def _initialize_reference_counters(self, nodes: t.List[Node]) -> None:
        """Perform initialization of reference counters for nodes in the allocation

        :param nodes: node attribute information for initializing the priorizer"""
        for node in nodes:
            # initialize all node counts to 0 and mark the entries "is_dirty=False"
            tracked = _TrackedNode(
                node.num_cpus, node.num_gpus, node.hostname, 0, set()
            )

            self._nodes[node.hostname] = tracked  # for O(1) access

            if node.num_gpus:
                self._gpu_refs.append(tracked)
            else:
                self._cpu_refs.append(tracked)

    # def _update_ref_count(self, host: str, updated_ref_count: _TrackedNode) -> None:
    #     """Updates the shared _NodeRefCount instance to keep each
    #     reference (cpu ref, gpu ref, all refs) in sync"""
    #     node = self._nodes[host]

    #     node.num_refs = updated_ref_count[0]
    #     node.dirty = updated_ref_count[2]

    def increment(
        self, host: str, tracking_id: t.Optional[str] = None
    ) -> NodeReferenceCount:
        """Directly increment the reference count of a given node and ensure the
        ref counter is marked as dirty to trigger a reordering on retrieval

        :param host: a hostname that should have a reference counter selected"""
        with self._lock:
            tracked_node = self._nodes[host]
            tracked_node.add(tracking_id)

            # self._update_ref_count(host, tracked_node)
            return tracked_node

    def _all_refs(self) -> t.List[_TrackedNode]:
        """Combine the CPU and GPU nodes into a single heap

        :returns: list of all reference counters"""
        refs = [*self._cpu_refs, *self._gpu_refs]
        heapq.heapify(refs)
        return refs

    def get_tracking_info(self, host: str) -> NodeReferenceCount:
        """Returns the reference counter information for a single node

        :param host: a hostname that should have a reference counter selected
        :returns: a reference counter"""
        return self._nodes[host]

    def decrement(
        self, host: str, tracking_id: t.Optional[str] = None
    ) -> NodeReferenceCount:
        """Directly increment the reference count of a given node and ensure the
        ref counter is marked as dirty to trigger a reordering

        :param host: a hostname that should have a reference counter decremented"""
        with self._lock:
            tracked_node = self._nodes[host]
            tracked_node.remove(tracking_id)

            # self._update_ref_count(host, tracked_node.as_refcount())
            return tracked_node

    def _create_sub_heap(self, hosts: t.List[str]) -> t.List[_TrackedNode]:
        """Create a new heap from the primary heap with user-specified nodes

        :param hosts: a list of hostnames used to filter the available nodes
        :returns: a list of assigned reference counters
        """
        nodes_tracking_info: t.List[_TrackedNode] = []

        # Collect all the tracking info for the requested nodes...
        for host in hosts:
            if tracking_info := self._nodes.get(host, None):
                nodes_tracking_info.append(tracking_info)

        # ... and use it to create a new heap from a specified subset of nodes
        heapq.heapify(nodes_tracking_info)

        return nodes_tracking_info

    def next_from(self, hosts: t.List[str]) -> t.Optional[Node]:
        """Return the next node available given a set of desired hosts

        :param hosts: a list of hostnames used to filter the available nodes
        :returns: a list of assigned reference counters
        :raises ValueError: if no host names are provided"""
        if not hosts or len(hosts) == 0:
            raise ValueError("No host names provided")

        sub_heap = self._create_sub_heap(hosts)
        return self._get_next_available_node(sub_heap)

    def next_n_from(self, num_items: int, hosts: t.List[str]) -> t.List[Node]:
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

    def unassigned(self, heap: t.Optional[t.List[_TrackedNode]] = None) -> t.List[Node]:
        """Select nodes that are currently not assigned a task

        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counts for all unassigned nodes"""
        if heap is None:
            heap = list(self._nodes.values())

        return list(filter(lambda x: x.num_refs == 0, heap))

    def assigned(self, heap: t.Optional[t.List[_TrackedNode]] = None) -> t.List[Node]:
        """Helper method to identify the nodes that are currently assigned

        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counts for all assigned nodes"""
        if heap is None:
            heap = list(self._nodes.values())

        return list(filter(lambda x: x.num_refs == 1, heap))

    def _check_satisfiable_n(
        self, num_items: int, heap: t.Optional[t.List[_TrackedNode]] = None
    ) -> bool:
        """Validates that a request for some number of nodes `n` can be
        satisfied by the prioritizer given the set of nodes available

        :param num_items: the desird number of nodes to allocate
        :param heap: (optional) a subset of the node heap to consider
        :returns: True if the request can be fulfilled, False otherwise"""
        num_nodes = len(self._nodes.keys())

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

    def _get_next_available_node(self, heap: t.List[_TrackedNode]) -> t.Optional[Node]:
        """Finds the next node w/the least amount of running processes and
        ensures that any elements that were directly updated are updated in
        the priority structure before being made available

        :param heap: (optional) a subset of the node heap to consider
        :returns: a reference counter for an available node if an unassigned node
        exists, `None` otherwise"""
        tracking_info: t.Optional[_TrackedNode] = None

        with self._lock:
            tracking_info = heapq.heappop(heap)
            is_dirty = tracking_info.dirty

            while is_dirty:
                if is_dirty:
                    # mark dirty items clean and place back into heap to be sorted
                    tracking_info.dirty = False
                    heapq.heappush(heap, tracking_info)

                tracking_info = heapq.heappop(heap)
                is_dirty = tracking_info.dirty

            original_ref_count = tracking_info.num_refs
            if original_ref_count == 0:
                # increment the ref count before putting back onto heap
                tracking_info.num_refs += 1

            heapq.heappush(heap, tracking_info)

            # next available must enforce only "open" return nodes
            if original_ref_count > 0:
                return None

        if not tracking_info:
            return None

        return tracking_info

    def _get_next_n_available_nodes(
        self,
        num_items: int,
        heap: t.List[_TrackedNode],
    ) -> t.List[Node]:
        """Find the next N available nodes w/least amount of references using
        the supplied filter to target a specific node capability

        :param n: number of nodes to reserve
        :returns: a list of reference counters for a available nodes if enough
        unassigned nodes exists, `None` otherwise
        :raises ValueError: if the number of requetsed nodes is not a positive integer
        """
        next_nodes: t.List[Node] = []

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

    def next(self, filter_on: t.Optional[PrioritizerFilter] = None) -> t.Optional[Node]:
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
    ) -> t.List[Node]:
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
