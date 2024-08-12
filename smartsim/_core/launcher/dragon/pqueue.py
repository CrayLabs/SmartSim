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

from smartsim.error.errors import SmartSimError
from smartsim.log import get_logger

logger = get_logger(__name__)


class Node(t.Protocol):
    """Base Node API required to support the NodePrioritizer"""

    @property
    def hostname(self) -> str:
        """The hostname of the node"""

    @property
    def num_cpus(self) -> int:
        """The number of CPUs in the node"""

    @property
    def num_gpus(self) -> int:
        """The number of GPUs in the node"""


class NodeReferenceCount(t.Protocol):
    """Contains details pertaining to references to a node"""

    @property
    def hostname(self) -> str:
        """The hostname of the node"""

    @property
    def num_refs(self) -> int:
        """The number of jobs assigned to the node"""


class _TrackedNode:
    """Node API required to have support in the NodePrioritizer"""

    def __init__(self, node: Node) -> None:
        self._node = node
        """The node being tracked"""
        self._num_refs = 0
        """The number of references to the tracked node"""
        self._assigned_tasks: t.Set[str] = set()
        """The unique identifiers of processes using this node"""
        self._is_dirty = False
        """Flag indicating that tracking information has been modified"""

    @property
    def hostname(self) -> str:
        """The hostname of the node being reference counted"""
        return self._node.hostname

    @property
    def num_cpus(self) -> int:
        """The number of CPUs of the node being reference counted"""
        return self._node.num_cpus

    @property
    def num_gpus(self) -> int:
        """The number of GPUs of the node being reference counted"""
        return self._node.num_gpus

    @property
    def num_refs(self) -> int:
        """The number of processes currently using the node"""
        return self._num_refs

    @property
    def is_assigned(self) -> int:
        """Returns True if no references are currently being counted"""
        return self._num_refs > 0

    @property
    def assigned_tasks(self) -> t.Set[str]:
        """The set of currently running processes of the node"""
        return self._assigned_tasks

    @property
    def is_dirty(self) -> bool:
        """The current modification status of the tracking information"""
        return self._is_dirty

    def clean(self) -> None:
        """Mark the node as unmodified"""
        self._is_dirty = False

    def add(
        self,
        tracking_id: t.Optional[str] = None,
    ) -> None:
        """Modify the node as needed to track the addition of a process

        :param tracking_id: (optional) a unique task identifier executing on the node
        to add"""
        if tracking_id in self.assigned_tasks:
            raise ValueError("Attempted adding task more than once")

        self._num_refs = self._num_refs + 1
        if tracking_id:
            self._assigned_tasks = self._assigned_tasks.union({tracking_id})
        self._is_dirty = True

    def remove(
        self,
        tracking_id: t.Optional[str] = None,
    ) -> None:
        """Modify the node as needed to track the removal of a process

        :param tracking_id: (optional) a unique task identifier executing on the node
        to remove"""
        if tracking_id and tracking_id not in self.assigned_tasks:
            raise ValueError("Attempted removal of untracked item")

        self._num_refs = max(self._num_refs - 1, 0)
        if tracking_id:
            self._assigned_tasks = self._assigned_tasks - {tracking_id}
        self._is_dirty = True

    def __lt__(self, other: "_TrackedNode") -> bool:
        """Comparison operator used to evaluate the ordering of nodes within
        the prioritizer. This comparison only considers reference counts.

        :param other: Another node to compare against"""
        if self.num_refs < other.num_refs:
            return True

        return False


class PrioritizerFilter(str, enum.Enum):
    """A filter used to select a subset of nodes to be queried"""

    CPU = enum.auto()
    GPU = enum.auto()


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
            # create a set of reference counters for the nodes
            tracked = _TrackedNode(node)

            self._nodes[node.hostname] = tracked  # for O(1) access

            if node.num_gpus:
                self._gpu_refs.append(tracked)
            else:
                self._cpu_refs.append(tracked)

    def increment(
        self, host: str, tracking_id: t.Optional[str] = None
    ) -> NodeReferenceCount:
        """Directly increment the reference count of a given node and ensure the
        ref counter is marked as dirty to trigger a reordering on retrieval

        :param host: a hostname that should have a reference counter selected
        :param tracking_id: (optional) a unique task identifier executing on the node
        to add"""
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
        :returns: a reference counter for the node
        :raises ValueError: if the hostname is not in the set of managed nodes"""
        if host not in self._nodes:
            raise ValueError("The supplied hostname was not found")

        return self._nodes[host]

    def decrement(
        self, host: str, tracking_id: t.Optional[str] = None
    ) -> NodeReferenceCount:
        """Directly increment the reference count of a given node and ensure the
        ref counter is marked as dirty to trigger a reordering

        :param host: a hostname that should have a reference counter decremented
        :param tracking_id: (optional) unique task identifier to remove"""
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

    def next_n_from(self, num_items: int, hosts: t.List[str]) -> t.Sequence[Node]:
        """Return the next N available nodes given a set of desired hosts

        :param num_items: the desired number of nodes to allocate
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
        self, heap: t.Optional[t.List[_TrackedNode]] = None
    ) -> t.Sequence[Node]:
        """Select nodes that are currently not assigned a task

        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counts for all unassigned nodes"""
        if heap is None:
            heap = list(self._nodes.values())

        nodes: t.List[_TrackedNode] = []
        for item in heap:
            if item.num_refs == 0:
                nodes.append(item)
        return nodes

    def assigned(
        self, heap: t.Optional[t.List[_TrackedNode]] = None
    ) -> t.Sequence[Node]:
        """Helper method to identify the nodes that are currently assigned

        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counts for all assigned nodes"""
        if heap is None:
            heap = list(self._nodes.values())

        nodes: t.List[_TrackedNode] = []
        for item in heap:
            if item.num_refs == 1:
                nodes.append(item)
        return nodes
        # return list(filter(lambda x: x.num_refs == 1, heap))

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

    def _get_next_available_node(
        self,
        heap: t.List[_TrackedNode],
        tracking_id: t.Optional[str] = None,
    ) -> t.Optional[Node]:
        """Finds the next node w/the least amount of running processes and
        ensures that any elements that were directly updated are updated in
        the priority structure before being made available

        :param heap: (optional) a subset of the node heap to consider
        :param tracking_id: (optional) unique task identifier to remove
        :returns: a reference counter for an available node if an unassigned node
        exists, `None` otherwise"""
        tracking_info: t.Optional[_TrackedNode] = None

        with self._lock:
            # re-sort the heap to handle any tracking changes
            if any(node.is_dirty for node in heap):
                heapq.heapify(heap)

            # grab the min node from the heap
            tracking_info = heapq.heappop(heap)

            # the node is available if it has no assigned tasks
            is_assigned = tracking_info.is_assigned
            if not is_assigned:
                # track the new process on the node
                tracking_info.add(tracking_id)

            # add the node that was popped back into the heap
            heapq.heappush(heap, tracking_info)

            # mark all nodes as clean now that everything is updated & sorted
            for node in heap:
                node.clean()

            # next available must only return previously unassigned nodes
            if is_assigned:
                return None

        return tracking_info

    def _get_next_n_available_nodes(
        self,
        num_items: int,
        heap: t.List[_TrackedNode],
        tracking_id: t.Optional[str] = None,
    ) -> t.List[Node]:
        """Find the next N available nodes w/least amount of references using
        the supplied filter to target a specific node capability

        :param num_items: number of nodes to reserve
        :param heap: (optional) a subset of the node heap to consider
        :returns: a list of reference counters for a available nodes if enough
        unassigned nodes exists, `None` otherwise
        :raises ValueError: if the number of requested nodes is not a positive integer
        """
        next_nodes: t.List[Node] = []

        if num_items < 1:
            raise ValueError(f"Number of items requested {num_items} is invalid")

        if not self._check_satisfiable_n(num_items, heap):
            return next_nodes

        while len(next_nodes) < num_items:
            next_node = self._get_next_available_node(heap, tracking_id)
            if next_node:
                next_nodes.append(next_node)
            else:
                break

        return next_nodes

    def _get_filtered_heap(
        self, filter_on: t.Optional[PrioritizerFilter] = None
    ) -> t.List[_TrackedNode]:
        """Helper method to select the set of nodes to include in a filtered
        heap.

        :param filter_on: A list of nodes that satisfy the filter. If no
        filter is supplied, all nodes are returned"""
        if filter_on == PrioritizerFilter.GPU:
            return self._gpu_refs
        if filter_on == PrioritizerFilter.CPU:
            return self._cpu_refs

        return self._all_refs()

    def next(self, filter_on: t.Optional[PrioritizerFilter] = None) -> t.Optional[Node]:
        """Find the next node available w/least amount of references using
        the supplied filter to target a specific node capability

        :param filter_on: the subset of nodes to query for available nodes
        :returns: a reference counter for an available node if an unassigned node
        exists, `None` otherwise"""
        heap = self._get_filtered_heap(filter_on)
        return self._get_next_available_node(heap)

    def next_n(
        self,
        num_items: int = 1,
        filter_on: t.Optional[PrioritizerFilter] = None,
        tracking_id: t.Optional[str] = None,
    ) -> t.List[Node]:
        """Find the next N available nodes w/least amount of references using
        the supplied filter to target a specific node capability

        :param num_items: number of nodes to reserve
        :param filter_on: the subset of nodes to query for available nodes
        :returns: Collection of reserved nodes"""
        heap = self._get_filtered_heap(filter_on)
        return self._get_next_n_available_nodes(num_items, heap, tracking_id)
