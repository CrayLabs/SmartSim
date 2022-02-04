# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

import sys
from ..config import CONFIG
from ..utils.helpers import create_lockfile_name


def write_colocated_launch_script(file_name, db_log, colocated_settings):
    """Write the colocated launch script

    This file will be written into the cwd of the step that
    is created for this entity.

    :param file_name: name of the script to write
    :type file_name: str
    :param db_log: log file for the db
    :type db_log: str
    :param colocated_settings: db settings from entity run_settings
    :type colocated_settings: dict[str, Any]
    """

    colocated_cmd = _build_colocated_wrapper_cmd(**colocated_settings,
                                                 db_log=db_log)

    with open(file_name, "w") as f:
        f.write("#!/bin/bash\n\n")

        f.write("export SMARTSIM_LOG_LEVEL=debug\n")
        f.write(f"{colocated_cmd}\n")
        f.write(f"DBPID=$!\n\n")
        if colocated_settings["limit_app_cpus"]:
            cpus = colocated_settings["cpus"]
            f.write(
                f"taskset -c 0-$(nproc --ignore={str(cpus+1)}) $@\n\n"
            )
        else:
            f.write(f"$@\n\n")

        f.write("if ps -p $DBPID > /dev/null\n")
        f.write("then\n")
        f.write("\tkill -15 $DBPID\n")
        f.write("fi\n")


def _build_colocated_wrapper_cmd(port=6780,
                                cpus=1,
                                interface="lo",
                                rai_args=None,
                                extra_db_args=None,
                                db_log=None,
                                **kwargs):
    """Build the command use to run a colocated db application

    :param port: db port, defaults to 6780
    :type port: int, optional
    :param cpus: db cpus, defaults to 1
    :type cpus: int, optional
    :param interface: network interface, defaults to "lo"
    :type interface: str, optional
    :param rai_args: redisai args, defaults to None
    :type rai_args: dict[str, str], optional
    :param extra_db_args: extra redis args, defaults to None
    :type extra_db_args: dict[str, str], optional
    :return: the command to run
    :rtype: str
    """

    # create unique lockfile name to avoid symlink vulnerability
    # this is the lockfile all the processes in the distributed
    # application will try to acquire. since we use a local tmp
    # directory on the compute node, only one process can acquire
    # the lock on the file.
    lockfile = create_lockfile_name()

    # create the command that will be used to launch the
    # database with the python entrypoint for starting
    # up the backgrounded db process
    cmd = [sys.executable,
           "-m",
           "smartsim._core.entrypoints.colocated",
           "+ifname",
           interface,
           "+lockfile",
           lockfile,
           "+db_cpus",
           str(cpus),
           "+command"
        ]

    # collect DB binaries and libraries from the config
    db_cmd = [
        CONFIG.redis_exe,
        CONFIG.redis_conf,
        "--loadmodule",
        CONFIG.redisai
    ]
    # add extra redisAI configurations
    for arg, value in rai_args.items():
        if value:
            # RAI wants arguments for inference in all capps
            # ex. THREADS_PER_QUEUE=1
            db_cmd.append(f"{arg.upper()} {str(value)}")

    # add port and log information
    db_cmd.extend([
        "--port",
        str(port),
        "--logfile",
        db_log
    ])
    for db_arg, value in extra_db_args.items():
        # replace "_" with "-" in the db_arg because we use kwargs
        # for the extra configurations and Python doesn't allow a hypon
        # in a variable name. All redis and KeyDB configuration options
        # use hyphens in their names.
        db_arg = db_arg.replace("_", "-")
        db_cmd.extend([
            f"--{db_arg}",
            value
        ])
    # run colocated db in the background
    db_cmd.append("&")

    cmd.extend(db_cmd)
    return " ".join(cmd)

