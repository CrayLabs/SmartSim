# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import typing as t


class ComputeNode:  # cov-slurm
    """The ComputeNode class holds resource information
    about a physical compute node
    """

    def __init__(
        self, node_name: t.Optional[str] = None, node_ppn: t.Optional[int] = None
    ) -> None:
        """Initialize a ComputeNode

        :param node_name: the name of the node
        :type node_name: str
        :param node_ppn: the number of ppn
        :type node_ppn: int
        """
        self.name: t.Optional[str] = node_name
        self.ppn: t.Optional[int] = node_ppn

    def _is_valid_node(self) -> bool:
        """Check if the node is complete

        Currently, validity is judged by name
        and ppn being not None.

        :returns: True if valid, false otherwise
        :rtype: bool
        """
        if self.name is None:
            return False
        if self.ppn is None:
            return False

        return True


class Partition:  # cov-slurm
    """The partition class holds information about
    a system partition.
    """

    def __init__(self) -> None:
        """Initialize a system partition"""
        self.name: t.Optional[str] = None
        self.min_ppn: t.Optional[int] = None
        self.nodes: t.Set[ComputeNode] = set()

    def _is_valid_partition(self) -> bool:
        """Check if the partition is valid

        Currently, validity is judged by name
        and each ComputeNode being valid

        :returns: True if valid, false otherwise
        :rtype: bool
        """
        if self.name is None:
            return False
        if len(self.nodes) <= 0:
            return False
        for node in self.nodes:
            if not node._is_valid_node():  # pylint: disable=protected-access
                return False

        return True
