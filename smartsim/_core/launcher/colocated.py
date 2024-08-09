# BSD 2-Clause License
#
# Copyright (c) 2021-2024 Hewlett Packard Enterprise
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


def colocated_settings():
    return None

def write_colocated_launch_script(file_name: str) -> None:
    """Write the colocated launch script

    This file will be written into the cwd of the step that
    is created for this entity.

    :param file_name: name of the script to write
    :param fs_log: log file for the fs
    :param colocated_settings: fs settings from entity run_settings
    """

    colocated_cmd = ""

    with open(file_name, "w", encoding="utf-8") as script_file:
        script_file.write("#!/bin/bash\n")
        script_file.write("set -e\n\n")

        script_file.write("Cleanup () {\n")
        script_file.write("if ps -p $DBPID > /dev/null; then\n")
        script_file.write("\tkill -15 $DBPID\n")
        script_file.write("fi\n}\n\n")

        # run cleanup after all exitcodes
        script_file.write("trap Cleanup exit\n\n")

        # force entrypoint to write some debug information to the
        # STDOUT of the job
        if colocated_settings["debug"]:
            script_file.write("export SMARTSIM_LOG_LEVEL=debug\n")
        script_file.write(f"db_stdout=$({colocated_cmd})\n")
        # extract and set DBPID within the shell script that is
        # enclosed between __PID__ and sent to stdout by the colocated
        # entrypoints file
        script_file.write(
            "DBPID=$(echo $db_stdout | sed -n "
            "'s/.*__PID__\\([0-9]*\\)__PID__.*/\\1/p')\n"
        )

        # Write the actual launch command for the app
        script_file.write("$@\n\n")
