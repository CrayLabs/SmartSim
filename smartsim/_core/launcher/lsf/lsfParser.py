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


def parse_bsub(output: str) -> str:
    """Parse bsub output and return job id.

    :param output: stdout of bsub command
    :type output: str
    :returns: job id
    :rtype: str
    """
    for line in output.split("\n"):
        if line.startswith("Job"):
            return line.split()[1][1:-1]
    return ""


def parse_bsub_error(output: str) -> str:
    """Parse and return error output of a failed bsub command.

    :param output: stderr of qsub command
    :type output: str
    :returns: error message
    :rtype: str
    """
    # Search for first non-empty line
    error_lines = []
    copy_lines = False
    for line in output.split("\n"):
        if line.strip().startswith("**"):
            copy_lines = True
        if line.strip().startswith("-----------"):
            copy_lines = False
        if copy_lines and line.lstrip("*").strip():
            error_lines += [line.lstrip("*").strip()]

    if error_lines:
        return "\n".join(error_lines).strip()

    # If the error message was not parsable, return it as is
    if output.strip():
        return output

    # If output is empty, present standard error
    base_err = "LSF run error"
    return base_err


def parse_jslist_stepid(output: str, step_id: str) -> t.Tuple[str, t.Optional[str]]:
    """Parse and return output of the jslist command run with
    options to obtain step status

    :param output: output of the bjobs command
    :type output: str
    :param step_id: allocation id or job step id
    :type step_id: str
    :return: status and return code
    :rtype: (str, str)
    """
    result: t.Tuple[str, t.Optional[str]] = ("NOTFOUND", None)

    for line in output.split("\n"):
        fields = line.split()
        if len(fields) >= 7:
            if fields[0] == step_id:
                stat = fields[6]
                return_code = fields[5]
                result = (stat, return_code)
                break
    return result


def parse_bjobs_jobid(output: str, job_id: str) -> str:
    """Parse and return output of the bjobs command run with options
    to obtain job status.

    :param output: output of the bjobs command
    :type output: str
    :param job_id: allocation id or job step id
    :type job_id: str
    :return: status
    :rtype: str
    """
    result = "NOTFOUND"
    for line in output.split("\n"):
        fields = line.split()
        if len(fields) >= 3:
            if fields[0] == job_id:
                stat = fields[2]
                result = stat
                break
    return result


def parse_bjobs_nodes(output: str) -> t.List[str]:
    """Parse and return the bjobs command run with
    options to obtain node list, i.e. with `-w`.

    This function parses and returns the nodes of
    a job in a list with the duplicates removed.

    :param output: output of the `bjobs -w` command
    :type output: str
    :return: compute nodes of the allocation or job
    :rtype: list of str
    """
    nodes = []

    lines = output.split("\n")
    nodes_str = lines[1].split()[5]
    nodes = nodes_str.split(":")

    return list(dict.fromkeys(nodes))


def parse_max_step_id_from_jslist(output: str) -> t.Optional[str]:
    """Parse and return the maximum step id from a jslist command.
    This function must be called immedietaly after a call to jsrun,
    and before the next one, to ensure the id of the last spawned task is
    properly returned

    :param output: output bjobs
    :type output: str
    :param step_name: the name of the step to query
    :type step_name: str
    :return: the step_id
    :rtype: str
    """
    max_step_id = None

    for line in output.split("\n"):
        if line.startswith("="):
            continue
        fields = line.split()
        if len(fields) >= 7:
            if fields[0].isdigit():
                if (max_step_id is None) or (int(fields[0]) > max_step_id):
                    max_step_id = int(fields[0])

    if max_step_id:
        return str(max_step_id)
    return None
