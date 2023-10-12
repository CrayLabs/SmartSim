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


def parse_cobalt_step_status(output: str, step_id: str) -> str:
    """
    Parse and return the status of a cobalt step

    :param output: output qstat
    :type output: str
    :param step_id: the id of the step to query
    :type step_id: str
    :rtype: str
    """
    status = "NOTFOUND"
    for line in output.split("\n"):
        fields = line.split()
        if len(fields) >= 2:
            if fields[0] == step_id:
                status = fields[1]
                break
    return status


def parse_cobalt_step_id(output: str, step_name: str) -> str:
    """Parse and return the step id from a cobalt qstat command

    :param output: output qstat
    :type output: str
    :param step_name: the name of the step to query
    :type step_name: str
    :return: the step_id
    :rtype: str
    """
    step_id = ""
    for line in output.split("\n"):
        fields = line.split()
        if len(fields) >= 2:
            if fields[0] == step_name:
                step_id = fields[1]
                break
    return step_id


def parse_qsub_out(output: str) -> str:
    """
    Parse and return the step id from a cobalt qsub command

    :param output: output qstat
    :type output: str
    :return: the step_id
    :rtype: str
    """
    step_id = ""
    for line in output.split("\n"):
        try:
            value = line.strip()
            int(value) # if the cast works, return original string
            step_id = value
            break
        except ValueError:
            continue
    return step_id
