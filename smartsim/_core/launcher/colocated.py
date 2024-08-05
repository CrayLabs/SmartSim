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

import sys
import typing as t

from ...entity.dbobject import FSModel, FSScript
from ...error import SSInternalError
from ..config import CONFIG
from ..utils.helpers import create_lockfile_name


def write_colocated_launch_script(
    file_name: str, fs_log: str, colocated_settings: t.Dict[str, t.Any]
) -> None:
    """Write the colocated launch script

    This file will be written into the cwd of the step that
    is created for this entity.

    :param file_name: name of the script to write
    :param fs_log: log file for the fs
    :param colocated_settings: fs settings from entity run_settings
    """

    colocated_cmd = _build_colocated_wrapper_cmd(fs_log, **colocated_settings)

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


def _build_colocated_wrapper_cmd(
    fs_log: str,
    cpus: int = 1,
    rai_args: t.Optional[t.Dict[str, str]] = None,
    extra_fs_args: t.Optional[t.Dict[str, str]] = None,
    port: int = 6780,
    ifname: t.Optional[t.Union[str, t.List[str]]] = None,
    custom_pinning: t.Optional[str] = None,
    **kwargs: t.Any,
) -> str:
    """Build the command use to run a colocated fs application

    :param fs_log: log file for the fs
    :param cpus: fs cpus
    :param rai_args: redisai args
    :param extra_fs_args: extra redis args
    :param port: port to bind fs to
    :param ifname: network interface(s) to bind fs to
    :param fs_cpu_list: The list of CPUs that the feature store should be limited to
    :return: the command to run
    """
    # pylint: disable=too-many-locals

    # create unique lockfile name to avoid symlink vulnerability
    # this is the lockfile all the processes in the distributed
    # application will try to acquire. since we use a local tmp
    # directory on the compute node, only one process can acquire
    # the lock on the file.
    lockfile = create_lockfile_name()

    # create the command that will be used to launch the
    # feature store with the python entrypoint for starting
    # up the backgrounded fs process

    cmd = [
        sys.executable,
        "-m",
        "smartsim._core.entrypoints.colocated",
        "+lockfile",
        lockfile,
        "+fs_cpus",
        str(cpus),
    ]
    # Add in the interface if using TCP/IP
    if ifname:
        if isinstance(ifname, str):
            ifname = [ifname]
        cmd.extend(["+ifname", ",".join(ifname)])
    cmd.append("+command")
    # collect fs binaries and libraries from the config

    fs_cmd = []
    if custom_pinning:
        fs_cmd.extend(["taskset", "-c", custom_pinning])
    fs_cmd.extend(
        [CONFIG.database_exe, CONFIG.database_conf, "--loadmodule", CONFIG.redisai]
    )

    # add extra redisAI configurations
    for arg, value in (rai_args or {}).items():
        if value:
            # RAI wants arguments for inference in all caps
            # ex. THREADS_PER_QUEUE=1
            fs_cmd.append(f"{arg.upper()} {str(value)}")

    fs_cmd.extend(["--port", str(port)])

    # Add socket and permissions for UDS
    unix_socket = kwargs.get("unix_socket", None)
    socket_permissions = kwargs.get("socket_permissions", None)

    if unix_socket and socket_permissions:
        fs_cmd.extend(
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

    fs_cmd.extend(
        ["--logfile", fs_log]
    )  # usually /dev/null, unless debug was specified
    if extra_fs_args:
        for fs_arg, value in extra_fs_args.items():
            # replace "_" with "-" in the fs_arg because we use kwargs
            # for the extra configurations and Python doesn't allow a hyphen
            # in a variable name. All redis and KeyDB configuration options
            # use hyphens in their names.
            fs_arg = fs_arg.replace("_", "-")
            fs_cmd.extend([f"--{fs_arg}", value])

    fs_models = kwargs.get("fs_models", None)
    if fs_models:
        fs_model_cmd = _build_fs_model_cmd(fs_models)
        fs_cmd.extend(fs_model_cmd)

    fs_scripts = kwargs.get("fs_scripts", None)
    if fs_scripts:
        fs_script_cmd = _build_fs_script_cmd(fs_scripts)
        fs_cmd.extend(fs_script_cmd)

    cmd.extend(fs_cmd)

    return " ".join(cmd)


def _build_fs_model_cmd(fs_models: t.List[FSModel]) -> t.List[str]:
    cmd = []
    for fs_model in fs_models:
        cmd.append("+fs_model")
        cmd.append(f"--name={fs_model.name}")

        # Here fs_model.file is guaranteed to exist
        # because we don't allow the user to pass a serialized FSModel
        cmd.append(f"--file={fs_model.file}")

        cmd.append(f"--backend={fs_model.backend}")
        cmd.append(f"--device={fs_model.device}")
        cmd.append(f"--devices_per_node={fs_model.devices_per_node}")
        cmd.append(f"--first_device={fs_model.first_device}")
        if fs_model.batch_size:
            cmd.append(f"--batch_size={fs_model.batch_size}")
        if fs_model.min_batch_size:
            cmd.append(f"--min_batch_size={fs_model.min_batch_size}")
        if fs_model.min_batch_timeout:
            cmd.append(f"--min_batch_timeout={fs_model.min_batch_timeout}")
        if fs_model.tag:
            cmd.append(f"--tag={fs_model.tag}")
        if fs_model.inputs:
            cmd.append("--inputs=" + ",".join(fs_model.inputs))
        if fs_model.outputs:
            cmd.append("--outputs=" + ",".join(fs_model.outputs))

    return cmd


def _build_fs_script_cmd(fs_scripts: t.List[FSScript]) -> t.List[str]:
    cmd = []
    for fs_script in fs_scripts:
        cmd.append("+fs_script")
        cmd.append(f"--name={fs_script.name}")
        if fs_script.func:
            # Notice that here fs_script.func is guaranteed to be a str
            # because we don't allow the user to pass a serialized function
            func = fs_script.func
            sanitized_func = func.replace("\n", "\\n")
            if not (
                sanitized_func.startswith("'")
                and sanitized_func.endswith("'")
                or (sanitized_func.startswith('"') and sanitized_func.endswith('"'))
            ):
                sanitized_func = '"' + sanitized_func + '"'
            cmd.append(f"--func={sanitized_func}")
        elif fs_script.file:
            cmd.append(f"--file={fs_script.file}")
        cmd.append(f"--device={fs_script.device}")
        cmd.append(f"--devices_per_node={fs_script.devices_per_node}")
        cmd.append(f"--first_device={fs_script.first_device}")
    return cmd
