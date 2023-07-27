# BSD 2-Clause License
#
# Copyright (c) 2021-2023 Hewlett Packard Enterprise
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
import typing as t


from ...error import SSInternalError
from ..config import CONFIG
from ..utils.helpers import create_lockfile_name
from ...entity.dbobject import DBModel, DBScript


def write_colocated_launch_script(
    file_name: str, db_log: str, colocated_settings: t.Dict[str, t.Any]
) -> None:
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

    colocated_cmd = _build_colocated_wrapper_cmd(db_log, **colocated_settings)

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

        script_file.write(f"{colocated_cmd}\n")
        script_file.write("DBPID=$!\n\n")

        # Write the actual launch command for the app
        script_file.write("$@\n\n")


def _build_colocated_wrapper_cmd(
    db_log: str,
    cpus: int = 1,
    rai_args: t.Optional[t.Dict[str, str]] = None,
    extra_db_args: t.Optional[t.Dict[str, str]] = None,
    port: int = 6780,
    ifname: t.Optional[t.Union[str, t.List[str]]] = None,
    custom_pinning: t.Optional[str] = None,
    **kwargs: t.Any,
) -> str:
    """Build the command use to run a colocated DB application

    :param db_log: log file for the db
    :type db_log: str
    :param cpus: db cpus, defaults to 1
    :type cpus: int, optional
    :param rai_args: redisai args, defaults to None
    :type rai_args: dict[str, str], optional
    :param extra_db_args: extra redis args, defaults to None
    :type extra_db_args: dict[str, str], optional
    :param port: port to bind DB to
    :type port: int
    :param ifname: network interface(s) to bind DB to
    :type ifname: str | list[str], optional
    :param db_cpu_list: The list of CPUs that the database should be limited to
    :type db_cpu_list: str, optional
    :return: the command to run
    :rtype: str
    """
    # pylint: disable=too-many-locals

    # create unique lockfile name to avoid symlink vulnerability
    # this is the lockfile all the processes in the distributed
    # application will try to acquire. since we use a local tmp
    # directory on the compute node, only one process can acquire
    # the lock on the file.
    lockfile = create_lockfile_name()

    # create the command that will be used to launch the
    # database with the python entrypoint for starting
    # up the backgrounded db process

    cmd = [
            sys.executable,
            "-m",
            "smartsim._core.entrypoints.colocated",
            "+lockfile",
            lockfile,
            "+db_cpus",
            str(cpus),
        ]
    # Add in the interface if using TCP/IP
    if ifname:
        if isinstance(ifname, str):
            ifname = [ifname]
        cmd.extend(["+ifname", ",".join(ifname)])
    cmd.append("+command")
    # collect DB binaries and libraries from the config

    db_cmd = []
    if custom_pinning:
        db_cmd.extend([
            'taskset', '-c', custom_pinning
        ])
    db_cmd.extend(
        [
            CONFIG.database_exe,
            CONFIG.database_conf,
            "--loadmodule",
            CONFIG.redisai
        ]
    )

    # add extra redisAI configurations
    for arg, value in (rai_args or {}).items():
        if value:
            # RAI wants arguments for inference in all caps
            # ex. THREADS_PER_QUEUE=1
            db_cmd.append(f"{arg.upper()} {str(value)}")

    db_cmd.extend(["--port", str(port)])

    # Add socket and permissions for UDS
    unix_socket = kwargs.get("unix_socket", None)
    socket_permissions = kwargs.get("socket_permissions", None)

    if unix_socket and socket_permissions:
        db_cmd.extend(
            [
                "--unixsocket",
                str(unix_socket),
                "--unixsocketperm",
                str(socket_permissions),
            ]
        )
    elif bool(unix_socket) ^ bool(socket_permissions):
        raise SSInternalError(
            "`unix_socket` and `socket_permissions` must both be defined or undefined."
        )

    db_cmd.extend(
        ["--logfile", db_log]
    )  # usually /dev/null, unless debug was specified
    if extra_db_args:
        for db_arg, value in extra_db_args.items():
            # replace "_" with "-" in the db_arg because we use kwargs
            # for the extra configurations and Python doesn't allow a hyphen
            # in a variable name. All redis and KeyDB configuration options
            # use hyphens in their names.
            db_arg = db_arg.replace("_", "-")
            db_cmd.extend([f"--{db_arg}", value])

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


def _build_db_model_cmd(db_models: t.List[DBModel]) -> t.List[str]:
    cmd = []
    for db_model in db_models:
        cmd.append("+db_model")
        cmd.append(f"--name={db_model.name}")

        # Here db_model.file is guaranteed to exist
        # because we don't allow the user to pass a serialized DBModel
        cmd.append(f"--file={db_model.file}")

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
            cmd.append("--inputs=" + ",".join(db_model.inputs))
        if db_model.outputs:
            cmd.append("--outputs=" + ",".join(db_model.outputs))

    return cmd


def _build_db_script_cmd(db_scripts: t.List[DBScript]) -> t.List[str]:
    cmd = []
    for db_script in db_scripts:
        cmd.append("+db_script")
        cmd.append(f"--name={db_script.name}")
        if db_script.func:
            # Notice that here db_script.func is guaranteed to be a str
            # because we don't allow the user to pass a serialized function
            sanitized_func = db_script.func.replace("\n", "\\n")
            if not (
                sanitized_func.startswith("'")
                and sanitized_func.endswith("'")
                or (sanitized_func.startswith('"') and sanitized_func.endswith('"'))
            ):
                sanitized_func = '"' + sanitized_func + '"'
            cmd.append(f"--func={sanitized_func}")
        elif db_script.file:
            cmd.append(f"--file={db_script.file}")
        cmd.append(f"--device={db_script.device}")
        cmd.append(f"--devices_per_node={db_script.devices_per_node}")

    return cmd
