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
from shutil import which

"""
Parsers for various slurm functions.
"""


def parse_salloc(output: str) -> t.Optional[str]:
    for line in output.split("\n"):
        if line.startswith("salloc: Granted job allocation"):
            return line.split()[-1]
    return None


def parse_salloc_error(output: str) -> t.Optional[str]:
    """Parse and return error output of a failed salloc command

    :param output: stderr output of salloc command
    :type output: str
    :return: error message
    :rtype: str
    """
    salloc = which("salloc")
    # look for error first
    for line in output.split("\n"):
        if salloc and line.startswith(salloc + ": error:"):
            error = line.split("error:")[1]
            return error.strip()
        if line.startswith("salloc: error:"):
            error = line.split("error:")[1]
            return error.strip()
    # if no error line, take first line
    for line in output.split("\n"):
        if salloc and line.startswith(salloc + ": "):
            error = " ".join((line.split()[1:]))
            return error.strip()
        if line.startswith("salloc: "):
            error = " ".join((line.split()[1:]))
            return error.strip()
    # return None if we cant find error
    return None


def jobid_exact_match(parsed_id: str, job_id: str) -> bool:
    """Check that job_id is an exact match and not
    the prefix of another job_id, like 1 and 11
    or 1.1 and 1.10. Works with job id or step
    id (i.e. with or without a '.' in the id)
    :param parsed_id: the id read from the line
    :type paserd_id: str
    :param job_id: the id to check for equality
    :type job_id: str
    """
    if "." in job_id:
        return parsed_id == job_id

    return parsed_id.split(".")[0] == job_id


def parse_sacct(output: str, job_id: str) -> t.Tuple[str, t.Optional[str]]:
    """Parse and return output of the sacct command

    :param output: output of the sacct command
    :type output: str
    :param job_id: allocation id or job step id
    :type job_id: str
    :return: status and returncode
    :rtype: tuple
    """
    result: t.Tuple[str, t.Optional[str]] = ("PENDING", None)
    for line in output.split("\n"):
        parts = line.split("|")
        if len(parts) >= 3:
            if jobid_exact_match(parts[0], job_id):
                stat = parts[1]
                return_code = parts[2].split(":")[0]
                result = (stat, return_code)
                break
    return result


def parse_sstat_nodes(output: str, job_id: str) -> t.List[str]:
    """Parse and return the sstat command

    This function parses and returns the nodes of
    a job in a list with the duplicates removed.

    :param output: output of the sstat command
    :type output: str
    :return: compute nodes of the allocation or job
    :rtype: list of str
    """
    nodes = []
    for line in output.split("\n"):
        sstat_string = line.split("|")

        # sometimes there are \n that we need to ignore
        if len(sstat_string) >= 2:
            if jobid_exact_match(sstat_string[0], job_id):
                node = sstat_string[1]
                nodes.append(node)
    return list(set(nodes))


def parse_step_id_from_sacct(output: str, step_name: str) -> t.Optional[str]:
    """Parse and return the step id from a sacct command

    :param output: output of sacct --noheader -p
                   --format=jobname,jobid --job <alloc>
    :type output: str
    :param step_name: the name of the step to query
    :type step_name: str
    :return: the step_id
    :rtype: str
    """
    step_id = None
    for line in output.split("\n"):
        sacct_string = line.split("|")
        if len(sacct_string) >= 2:
            if sacct_string[0] == step_name:
                step_id = sacct_string[1]
    return step_id
