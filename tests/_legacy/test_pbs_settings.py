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

from smartsim.error import SSConfigError
from smartsim.settings import QsubBatchSettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


def test_node_formatting():
    def validate_settings(settings, spec, num_nodes, num_cpus):
        assert settings._create_resource_list() == [
            f"-l {spec}={num_nodes}:ncpus={num_cpus}"
        ]
        assert settings._ncpus == num_cpus

    num_nodes = 10
    num_cpus = 36

    # Test by specifying the number of nodes via setting a resource
    for spec in ["nodes", "select"]:
        # Test by setting nodes
        settings = QsubBatchSettings()
        settings.set_resource(spec, num_nodes)
        settings.set_ncpus(36)
        validate_settings(settings, spec, num_nodes, num_cpus)

    # Test when setting nodes through the constructor
    settings = QsubBatchSettings(ncpus=num_cpus, nodes=num_nodes)
    validate_settings(settings, "nodes", num_nodes, num_cpus)

    # Test when setting nodes through the constructor via resource
    settings = QsubBatchSettings(ncpus=num_cpus, resources={"nodes": num_nodes})
    validate_settings(settings, "nodes", num_nodes, num_cpus)

    # Test when setting select through the constructor via resource
    settings = QsubBatchSettings(ncpus=num_cpus, resources={"select": num_nodes})
    validate_settings(settings, "select", num_nodes, num_cpus)


def test_select_nodes_error():
    # # Test failure on initialization
    with pytest.raises(SSConfigError):
        QsubBatchSettings(nodes=10, resources={"select": 10})

    # Test setting via nodes and then select
    settings = QsubBatchSettings()
    settings.set_nodes(10)
    with pytest.raises(SSConfigError):
        settings.set_resource("select", 10)

    # Manually put "select" in the resource dictionary and
    # make sure the resource formatter catches the error
    settings = QsubBatchSettings()
    with pytest.raises(SSConfigError):
        settings.resources = {"nodes": 10, "select": 20}

    # # Test setting via select and then nodes
    settings = QsubBatchSettings()
    settings.set_resource("select", 10)
    with pytest.raises(SSConfigError):
        settings.set_nodes(10)


def test_resources_is_a_copy():
    settings = QsubBatchSettings()
    resources = settings.resources
    assert resources is not settings._resources


def test_nodes_and_select_not_ints_error():
    expected_error = TypeError
    with pytest.raises(expected_error):
        settings = QsubBatchSettings()
        settings.set_nodes("10")
    with pytest.raises(expected_error):
        settings = QsubBatchSettings()
        settings.set_resource("nodes", "10")
    with pytest.raises(expected_error):
        settings = QsubBatchSettings()
        settings.set_resource("select", "10")
    with pytest.raises(expected_error):
        settings = QsubBatchSettings()
        settings.resources = {"nodes": "10"}
    with pytest.raises(expected_error):
        settings = QsubBatchSettings()
        settings.resources = {"select": "10"}


def test_resources_not_set_on_error():
    settings = QsubBatchSettings(nodes=10)
    unaltered_resources = settings.resources
    with pytest.raises(SSConfigError):
        settings.resources = {"nodes": 10, "select": 10}

    assert unaltered_resources == settings.resources


def test_valid_types_in_resources():
    settings = QsubBatchSettings(nodes=10)
    with pytest.raises(TypeError):
        settings.set_resource("foo", None)
