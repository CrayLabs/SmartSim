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
from ...error import SSUnsupportedError
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
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")

        f.write("Cleanup () {\n")
        f.write("if ps -p $DBPID > /dev/null; then\n")
        f.write("\tkill -15 $DBPID\n")
        f.write("fi\n}\n\n")

        # run cleanup after all exitcodes
        f.write("trap Cleanup exit\n\n")

        # force entrypoint to write some debug information to the
        # STDOUT of the job
        if colocated_settings["debug"]:
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
        CONFIG.database_exe,
        CONFIG.database_conf,
        "--loadmodule",
        CONFIG.redisai
    ]
    # add extra redisAI configurations
    for arg, value in rai_args.items():
        if value:
            # RAI wants arguments for inference in all caps
            # ex. THREADS_PER_QUEUE=1
            db_cmd.append(f"{arg.upper()} {str(value)}")

    # add port and log information
    db_cmd.extend([
        "--port",
        str(port),
        "--logfile",
        db_log # usually /dev/null
    ])
    for db_arg, value in extra_db_args.items():
        # replace "_" with "-" in the db_arg because we use kwargs
        # for the extra configurations and Python doesn't allow a hyphen
        # in a variable name. All redis and KeyDB configuration options
        # use hyphens in their names.
        db_arg = db_arg.replace("_", "-")
        db_cmd.extend([
            f"--{db_arg}",
            value
        ])

    db_models = kwargs.get("db_models", None)
    if db_models:
        db_model_cmd = _build_db_model_cmd(db_models)
        db_cmd.extend(db_model_cmd)
    
    db_scripts = kwargs.get("db_scripts", None)
    if db_scripts:
        db_script_cmd = _build_db_script_cmd(db_scripts)
        db_cmd.extend(db_script_cmd)

    # run colocated db in the background
    db_cmd.append("&")

    cmd.extend(db_cmd)
    return " ".join(cmd)


def _build_db_model_cmd(db_models):
    cmd = []
    for db_model in db_models:
        cmd.append("+db_model")
        cmd.append(f"--name={db_model.name}")
        if db_model.file:
            cmd.append(f"--file={db_model.file}")
        else:
            # This is caught when the DBModel is added through add_ml_model,
            # but we keep this check for the sake of safety in case
            # DBModels are just copied over from another entity
            err_msg = "ML model can not be set from memory for colocated databases.\n"
            err_msg += "Please store the ML model in binary format "
            err_msg += "and add it to the SmartSim Model as file."
            raise SSUnsupportedError(err_msg)
        cmd.append(f"--backend={db_model.backend}")
        cmd.append(f"--device={db_model.device}")
        cmd.append(f"--devices_per_node={db_model.devices_per_node}")
        if db_model.batch_size:
            cmd.append(f"--batch_size={db_model.batch_size}")
        if db_model.min_batch_size:
            cmd.append(f"--min_batch_size={db_model.min_batch_size}")
        if db_model.min_batch_timeout:
            cmd.append(f"--min_batch_timeout={db_model.min_batch_timeout}")
        if db_model.tag:
            cmd.append(f"--tag={db_model.tag}")
        if db_model.inputs:
            cmd.append("--inputs="+",".join(db_model.inputs))
        if db_model.outputs:
            cmd.append("--outputs="+",".join(db_model.outputs))

    return cmd


def _build_db_script_cmd(db_scripts):
    cmd = []
    for db_script in db_scripts:
        cmd.append("+db_script")
        cmd.append(f"--name={db_script.name}")
        if db_script.func:
            # This is caught when the DBScript is added through add_script,
            # but we keep this check for the sake of safety in case
            # DBScripts are just copied over from another entity
            if not isinstance(db_script.func, str):
                err_msg = "Functions can not be set from memory for colocated databases.\n"
                err_msg += "Please convert the function to a string or store it as a text file "
                err_msg += "and add it to the SmartSim Model with add_script."
                raise SSUnsupportedError(err_msg)

            sanitized_func = db_script.func.replace("\n", "\\n")
            if not (sanitized_func.startswith("'") and sanitized_func.endswith("'")
               or (sanitized_func.startswith('"') and sanitized_func.endswith('"'))):
               sanitized_func = "\"" + sanitized_func + "\""
            cmd.append(f"--func={sanitized_func}")
        elif db_script.file:
            cmd.append(f"--file={db_script.file}")
        cmd.append(f"--device={db_script.device}")
        cmd.append(f"--devices_per_node={db_script.devices_per_node}")

    return cmd
        