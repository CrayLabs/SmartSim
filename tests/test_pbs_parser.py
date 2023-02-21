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

from os.path import dirname
from pathlib import Path

from smartsim._core.launcher.pbs import pbsParser

# -- qsub ---------------------------------------------------------


def test_parse_qsub():
    output = "12345.sdb"
    step_id = pbsParser.parse_qsub(output)
    assert step_id == "12345.sdb"


def test_parse_qsub_error():
    output = "qsub: Unknown queue"
    error = "Unknown queue"
    parsed_error = pbsParser.parse_qsub_error(output)
    assert error == parsed_error


# -- qstat ---------------------------------------------------------


def test_parse_qstat_nodes(fileutils):
    """Parse nodes from qsub called with -f -F json"""
    file_path = fileutils.get_test_conf_path("qstat.json")
    output = Path(file_path).read_text()
    nodes = ["server_1", "server_2"]
    parsed_nodes = pbsParser.parse_qstat_nodes(output)
    assert nodes == parsed_nodes


def test_parse_qstat_status():
    """test retrieval of status and exitcode"""
    output = (
        "Job id            Name             User              Time Use S Queue\n"
        "----------------  ---------------- ----------------  -------- - -----\n"
        "1289903.sdb       jobname          username          00:00:00 R queue\n"
    )
    status = "R"
    parsed_status = pbsParser.parse_qstat_jobid(output, "1289903.sdb")
    assert status == parsed_status
