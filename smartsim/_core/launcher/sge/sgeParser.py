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

import itertools
import json
import typing as t
import xml.etree.ElementTree as ET


def parse_qsub(output: str) -> str:
    """Parse qsub output and return job id. For SGE, the
    output is the job id itself.

    :param output: stdout of qsub command
    :returns: job id
    """
    return output


def parse_qsub_error(output: str) -> str:
    """Parse and return error output of a failed qsub command.

    :param output: stderr of qsub command
    :returns: error message
    """
    # look for error first
    for line in output.split("\n"):
        if line.startswith("qsub:"):
            error = line.split(":")[1]
            return error.strip()
    # if no error line, take first line
    for line in output.split("\n"):
        return line.strip()
    # if neither, present a base error message
    base_err = "PBS run error"
    return base_err


# def parse_qstat_jobid(output: str, job_id: str) -> t.Optional[str]:
#     """Parse and return output of the qstat command run with options
#     to obtain job status.
#
#     :param output: output of the qstat command
#     :param job_id: allocation id or job step id
#     :return: status
#     """
#     result = None
#     for line in output.split("\n"):
#         fields = line.split()
#         if len(fields) >= 5:
#             if fields[0] == job_id:
#                 stat = fields[4]
#                 result = stat
#                 break
#     return result


def parse_qstat_jobid_xml(output: str, job_id: str) -> t.Optional[str]:
    """Parse and return output of the qstat command run with XML options
    to obtain job status.

    :param output: output of the qstat command in JSON format
    :param job_id: allocation id or job step id
    :return: status
    """

    root = ET.fromstring(output)
    for job_list in root.findall(".//job_list"):
        job_state = job_list.find("state")
        # not None construct is needed here, since element with no
        # children returns 0, interpreted as False
        if (job_number := job_list.find("JB_job_number")) is not None:
            if job_number.text == job_id and (job_state is not None):
                return job_state.text

    return None


def parse_qacct_job_output(output: str, field_name: str) -> t.Union[str, int]:
    """Parse the output from qacct for a single job

    :param output: The raw text output from qacct
    :param field_name: The name of the field to extract
    """

    for line in output.splitlines():
        if field_name in line:
            return line.split()[1]

    return 1


def parse_step_id_from_qstat(output: str, step_name: str) -> t.Optional[str]:
    """Parse and return the step id from a qstat command

    :param output: output qstat
    :param step_name: the name of the step to query
    :return: the step_id
    """
    step_id: t.Optional[str] = None
    out_json = load_and_clean_json(output)

    if "Jobs" not in out_json:
        return step_id
    jobs = out_json["Jobs"]
    for key, val in jobs.items():
        if val["Job_Name"] == step_name:
            step_id = key
            return step_id

    return step_id


def load_and_clean_json(out: str) -> t.Any:
    if len(out.strip()) == 0:
        return ""
    try:
        json_out = json.loads(out)
        return json_out
    except json.decoder.JSONDecodeError as e:
        out_arr = out.split("\n")
        new_out = ""
        for lineno, line in enumerate(out_arr):
            if lineno != e.lineno - 1:
                new_out = new_out + "\n" + line
        return load_and_clean_json(new_out)
