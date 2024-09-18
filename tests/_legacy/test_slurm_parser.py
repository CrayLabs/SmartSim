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

from smartsim._core.launcher.slurm import slurm_parser

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


# -- Salloc ---------------------------------------------------------


def test_parse_salloc():
    output = (
        "salloc: Granted job allocation 118568\n"
        "salloc: Waiting for resource configuration\n"
        "salloc: Nodes nid00116 are ready for job"
    )
    alloc_id = slurm_parser.parse_salloc(output)
    assert alloc_id == "118568"


def test_parse_salloc_extra():
    output = (
        "salloc: Running node verification test prior to job execution.\n"
        "salloc: Will use xperf arguments in SLURM_SUBMIT_DIR/xperf-args if it exists.\n"
        "salloc: Results saved to SLURM_SUBMIT_DIR/nodeverify.jobid\n"
        "\nsalloc: Granted job allocation 22942\n"
        "salloc: Waiting for resource configuration\n"
        "salloc: Nodes prod76-0006 are ready for job\n"
    )
    alloc_id = slurm_parser.parse_salloc(output)
    assert alloc_id == "22942"


def test_parse_salloc_high():
    output = (
        "salloc: Granted job allocation 29917893\n"
        "salloc: Waiting for resource configuration\n"
        "salloc: Nodes nid00034 are ready for job\n"
    )
    alloc_id = slurm_parser.parse_salloc(output)
    assert alloc_id == "29917893"


def test_parse_salloc_error():
    output = "salloc: error: Job submit/allocate failed: Job dependency problem"
    error = "Job submit/allocate failed: Job dependency problem"
    parsed_error = slurm_parser.parse_salloc_error(output)
    assert error == parsed_error


def test_parse_salloc_error_2():
    output = (
        "salloc: unrecognized option '--no-a-option'\n"
        "Try 'salloc --help' for more information\n"
    )
    error = "unrecognized option '--no-a-option'"
    parsed_error = slurm_parser.parse_salloc_error(output)
    assert error == parsed_error


def test_parse_salloc_error_3():
    output = (
        "salloc: Running node verification test prior to job execution.\n"
        "salloc: Will use xperf arguments in SLURM_SUBMIT_DIR/xperf-args if it exists.\n"
        "salloc: Results saved to SLURM_SUBMIT_DIR/nodeverify.jobid\n"
        "\nsalloc: error: Job submit/allocate failed: Invalid node name specified\n"
    )
    error = "Job submit/allocate failed: Invalid node name specified"
    parsed_error = slurm_parser.parse_salloc_error(output)
    assert error == parsed_error


def test_parse_salloc_error_4():
    output = (
        "salloc: error: No hardware architecture specified (-C)!\n"
        "salloc: error: Job submit/allocate failed: Unspecified error\n"
    )
    error = "No hardware architecture specified (-C)!"
    parsed_error = slurm_parser.parse_salloc_error(output)
    assert error == parsed_error


# -- sstat ---------------------------------------------------------


def test_parse_sstat_nodes():
    """Parse nodes from sstat called with args -i -a -p -n
    PrologFlags=Alloc, Contain
    """
    output = "118594.extern|nid00028|38671|\n" "118594.0|nid00028|38703|"
    nodes = ["nid00028"]
    parsed_nodes = slurm_parser.parse_sstat_nodes(output, "118594")
    assert nodes == parsed_nodes


def test_parse_sstat_nodes_1():
    """Parse nodes from sstat called with args -i -a -p -n
    PrologFlags=Alloc
    """
    output = "22942.0|prod76-0006|354345|"
    nodes = ["prod76-0006"]
    parsed_nodes = slurm_parser.parse_sstat_nodes(output, "22942.0")
    assert nodes == parsed_nodes


def test_parse_sstat_nodes_2():
    """Parse nodes from sstat called with args -i -a -p -n
    PrologFlags=Alloc,Contain
    """
    output = "29917893.extern|nid00034|44860|\n" "29917893.0|nid00034|44887|\n"
    nodes = ["nid00034"]
    parsed_nodes = slurm_parser.parse_sstat_nodes(output, "29917893.0")
    assert nodes == parsed_nodes


def test_parse_sstat_nodes_3():
    """Parse nodes from sstat called with args -i -a -p -n
    Special case where interactive queue also causes there
    to be a constantly running .0 job
    PrologFlags=Alloc,Contain
    """
    output = (
        "29917893.extern|nid00034|44860|\n"
        "29917893.0|nid00034|44887,45151,45152,45153,45154,45155|\n"
        "29917893.2|nid00034|45174|\n"
    )
    nodes = ["nid00034"]
    parsed_nodes = slurm_parser.parse_sstat_nodes(output, "29917893.2")
    assert nodes == parsed_nodes


def test_parse_sstat_nodes_4():
    """Parse nodes from sstat called with args -i -a -p -n

    with extra steps

    PrologFlags=Alloc,Contain
    """
    output = (
        "30000.extern|nid00034|44860|\n"
        "30000.batch|nid00034|42352"
        "30000.0|nid00034|44887,45151,45152,45153,45154,45155|\n"
        "30000.1|nid00035|45174|\n"
        "30000.2|nid00036|45174,32435|\n"
    )
    nodes = set(["nid00034", "nid00035", "nid00036"])
    parsed_nodes = set(slurm_parser.parse_sstat_nodes(output, "30000"))
    assert nodes == parsed_nodes


def test_parse_sstat_nodes_4():
    """Parse nodes from sstat called with args -i -a -p -n

    with extra steps

    PrologFlags=Alloc,Contain
    """
    output = (
        "30000.extern|nid00034|44860|\n"
        "30000.batch|nid00034|42352"
        "30000.0|nid00034|44887,45151,45152,45153,45154,45155|\n"
        "30000.1|nid00035|45174|\n"
        "30000.2|nid00036|45174,32435|\n"
    )
    nodes = set(["nid00034", "nid00035", "nid00036"])
    parsed_nodes = set(slurm_parser.parse_sstat_nodes(output, "30000"))
    assert nodes == parsed_nodes


def test_parse_sstat_nodes_5():
    """Parse nodes from sstat called with args -i -a -p -n
    Special case where interactive queue also causes there
    to be a constantly running .0 job
    PrologFlags=Alloc,Contain
    """
    output = (
        "29917893.extern|nid00034|44860|\n"
        "29917893.20|nid00034|44887,45151,45152,45153,45154,45155|\n"
        "29917893.2|nid00034|45174|\n"
    )
    nodes = ["nid00034"]
    parsed_nodes = slurm_parser.parse_sstat_nodes(output, "29917893.2")
    assert nodes == parsed_nodes


# -- sacct ---------------------------------------------------------


def test_parse_sacct_step_id():
    output = (
        "SmartSim|119225|\n"
        "extern|119225.extern|\n"
        "m1-119225.0|119225.0|\n"
        "m2-119225.1|119225.1|"
    )
    step_id = "119225.0"
    parsed_step_id = slurm_parser.parse_step_id_from_sacct(output, "m1-119225.0")
    assert step_id == parsed_step_id


def test_parse_sacct_step_id_2():
    output = (
        "SmartSim|119225|\n"
        "extern|119225.extern|\n"
        "m1-119225.0|119225.0|\n"
        "m2-119225.1|119225.1|\n"
        "featurestore_0-119225.2|119225.2|\n"
        "n1-119225.3|119225.3|"
    )
    step_id = "119225.2"
    parsed_step_id = slurm_parser.parse_step_id_from_sacct(
        output, "featurestore_0-119225.2"
    )
    assert step_id == parsed_step_id


def test_parse_sacct_step_id_2():
    output = (
        "SmartSim|962333|\n"
        "extern|962333.extern|\n"
        "python-962333.0|962333.0|\n"
        "python-962333.1|962333.1|\n"
        "cti_dlaunch1.0|962333.2|\n"
        "cti_dlaunch1.0|962333.3|"
    )
    step_id = "962333.1"
    parsed_step_id = slurm_parser.parse_step_id_from_sacct(output, "python-962333.1")
    assert step_id == parsed_step_id


def test_parse_sacct_status():
    """test retrieval of status and exitcode
    PrologFlags=Alloc,Contain
    """
    output = "29917893.2|COMPLETED|0:0|\n"
    status = ("COMPLETED", "0")
    parsed_status = slurm_parser.parse_sacct(output, "29917893.2")
    assert status == parsed_status


def test_parse_sacct_status_1():
    """test retrieval of status and exitcode
    PrologFlags=Alloc
    """
    output = "22999.0|FAILED|1:0|\n"
    status = ("FAILED", "1")
    parsed_status = slurm_parser.parse_sacct(output, "22999.0")
    assert status == parsed_status


def test_parse_sacct_status_2():
    """test retrieval of status and exitcode
    PrologFlags=Alloc
    """
    output = "22999.10|COMPLETED|0:0|\n22999.1|FAILED|1:0|\n"
    status = ("FAILED", "1")
    parsed_status = slurm_parser.parse_sacct(output, "22999.1")
    assert status == parsed_status
