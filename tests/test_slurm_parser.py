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

import textwrap

import pytest

from smartsim._core.launcher_.slurm import slurm_parser

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


# -- Salloc ---------------------------------------------------------


@pytest.mark.parametrize(
    "output, expected_id",
    (
        pytest.param(
            textwrap.dedent("""\
                salloc: Granted job allocation 118568
                salloc: Waiting for resource configuration
                salloc: Nodes nid00116 are ready for job
                """),
            "118568",
            id="Simple",
        ),
        pytest.param(
            textwrap.dedent("""\
                salloc: Running node verification test prior to job execution.
                salloc: Will use xperf arguments in SLURM_SUBMIT_DIR/xperf-args if it exists.
                salloc: Results saved to SLURM_SUBMIT_DIR/nodeverify.jobid
                
                salloc: Granted job allocation 22942
                salloc: Waiting for resource configuration
                salloc: Nodes prod76-0006 are ready for job
                """),
            "22942",
            id="Extra",
        ),
        pytest.param(
            textwrap.dedent("""\
                salloc: Granted job allocation 29917893
                salloc: Waiting for resource configuration
                salloc: Nodes nid00034 are ready for job
                """),
            "29917893",
            id="High",
        ),
    ),
)
def test_parse_salloc(output, expected_id):
    alloc_id = slurm_parser.parse_salloc(output)
    assert alloc_id == expected_id


@pytest.mark.parametrize(
    "output, expected_error",
    (
        pytest.param(
            "salloc: error: Job submit/allocate failed: Job dependency problem",
            "Job submit/allocate failed: Job dependency problem",
            id="Dependency Problem",
        ),
        pytest.param(
            textwrap.dedent("""\
                salloc: unrecognized option '--no-a-option'
                Try 'salloc --help' for more information
                """),
            "unrecognized option '--no-a-option'",
            id="Bad Option",
        ),
        pytest.param(
            textwrap.dedent("""\
                salloc: Running node verification test prior to job execution.
                salloc: Will use xperf arguments in SLURM_SUBMIT_DIR/xperf-args if it exists.
                salloc: Results saved to SLURM_SUBMIT_DIR/nodeverify.jobid

                salloc: error: Job submit/allocate failed: Invalid node name specified
                """),
            "Job submit/allocate failed: Invalid node name specified",
            id="Bad Node Name",
        ),
        pytest.param(
            textwrap.dedent("""\
                salloc: error: No hardware architecture specified (-C)!
                salloc: error: Job submit/allocate failed: Unspecified error
                """),
            "No hardware architecture specified (-C)!",
            id="Missing HW Architecture",
        ),
    ),
)
def test_parse_salloc_error(output, expected_error):
    parsed_error = slurm_parser.parse_salloc_error(output)
    assert expected_error == parsed_error


# -- sstat ---------------------------------------------------------


@pytest.mark.parametrize(
    "output, job_id, nodes",
    (
        pytest.param(
            textwrap.dedent("""\
                118594.extern|nid00028|38671|
                118594.0|nid00028|38703|"""),
            "118594",
            {"nid00028"},
            id="No suffix",
        ),
        pytest.param(
            "22942.0|prod76-0006|354345|", "22942.0", {"prod76-0006"}, id="with suffix"
        ),
        pytest.param(
            textwrap.dedent("""\
                29917893.extern|nid00034|44860|
                29917893.0|nid00034|44887|
                """),
            "29917893.0",
            {"nid00034"},
            id="with suffix and extern",
        ),
        pytest.param(
            textwrap.dedent("""\
                29917893.extern|nid00034|44860|
                29917893.0|nid00034|44887,45151,45152,45153,45154,45155|
                29917893.2|nid00034|45174|
                """),
            "29917893.2",
            {"nid00034"},
            id="With interactive queue runnning `.0` job",
        ),
        pytest.param(
            textwrap.dedent("""\
                30000.extern|nid00034|44860|
                30000.batch|nid00034|42352
                30000.0|nid00034|44887,45151,45152,45153,45154,45155|
                30000.1|nid00035|45174|
                30000.2|nid00036|45174,32435|
                """),
            "30000",
            {"nid00034", "nid00035", "nid00036"},
            id="With extra steps",
        ),
        pytest.param(
            textwrap.dedent("""\
                29917893.extern|nid00034|44860|
                29917893.20|nid00034|44887,45151,45152,45153,45154,45155|
                29917893.2|nid00034|45174|
                """),
            "29917893.2",
            {"nid00034"},
            id="with suffix and lines with same prefix",
        ),
    ),
)
def test_parse_sstat_nodes(output, job_id, nodes):
    """Parse nodes from sstat called with args -i -a -p -n
    PrologFlags=Alloc, Contain
    """
    parsed_nodes = slurm_parser.parse_sstat_nodes(output, job_id)
    assert len(nodes) == len(parsed_nodes)
    assert nodes == set(parsed_nodes)


# -- sacct ---------------------------------------------------------


@pytest.mark.parametrize(
    "output, step_name, step_id",
    (
        pytest.param(
            textwrap.dedent("""\
                SmartSim|119225|
                extern|119225.extern|
                m1-119225.0|119225.0|
                m2-119225.1|119225.1|"""),
            "m1-119225.0",
            "119225.0",
            id="Basic",
        ),
        pytest.param(
            textwrap.dedent("""\
                SmartSim|119225|
                extern|119225.extern|
                m1-119225.0|119225.0|
                m2-119225.1|119225.1|
                featurestore_0-119225.2|119225.2|
                n1-119225.3|119225.3|"""),
            "featurestore_0-119225.2",
            "119225.2",
            id="New Job",
        ),
        pytest.param(
            textwrap.dedent("""\
                SmartSim|962333|
                extern|962333.extern|
                python-962333.0|962333.0|
                python-962333.1|962333.1|
                cti_dlaunch1.0|962333.2|
                cti_dlaunch1.0|962333.3|"""),
            "python-962333.1",
            "962333.1",
            id="Very similar names",
        ),
    ),
)
def test_parse_sacct_step_id(output, step_name, step_id):
    parsed_step_id = slurm_parser.parse_step_id_from_sacct(output, step_name)
    assert step_id == parsed_step_id


@pytest.mark.parametrize(
    "output, job_id, status",
    (
        pytest.param(
            "29917893.2|COMPLETED|0:0|\n",
            "29917893.2",
            ("COMPLETED", "0"),
            id="Completed",
        ),
        pytest.param(
            "22999.0|FAILED|1:0|\n",
            "22999.0",
            ("FAILED", "1"),
            id="Failed",
        ),
        pytest.param(
            textwrap.dedent("""\
                22999.10|COMPLETED|0:0|
                22999.1|FAILED|1:0|
                """),
            "22999.1",
            ("FAILED", "1"),
            id="Failed with extra",
        ),
    ),
)
def test_parse_sacct_status(output, job_id, status):
    """test retrieval of status and exitcode
    PrologFlags=Alloc,Contain
    """
    parsed_status = slurm_parser.parse_sacct(output, job_id)
    assert status == parsed_status
