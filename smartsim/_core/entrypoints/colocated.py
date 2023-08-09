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

import argparse
import os
import signal
import socket
import sys
import tempfile
import typing as t

from pathlib import Path
from subprocess import PIPE, STDOUT
from types import FrameType


import filelock
import psutil
from smartredis import Client
from smartredis.error import RedisConnectionError, RedisReplyError

from smartsim._core.utils.network import current_ip
from smartsim.error import SSInternalError
from smartsim.log import get_logger

logger = get_logger(__name__)

DBPID = None

# kill is not catchable
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGABRT]


def handle_signal(signo: int, _frame: t.Optional[FrameType]) -> None:
    if not signo:
        logger.warning("Received signal with no signo")
    cleanup()


def launch_db_model(client: Client, db_model: t.List[str]) -> str:
    """Parse options to launch model on local cluster

    :param client: SmartRedis client connected to local DB
    :type client: Client
    :param db_model: List of arguments defining the model
    :type db_model: List[str]
    :return: Name of model
    :rtype: str
    """
    parser = argparse.ArgumentParser("Set ML model on DB")
    parser.add_argument("--name", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--backend", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--devices_per_node", type=int)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--min_batch_size", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--outputs", nargs="+", default=None)

    # Unused if we use SmartRedis
    parser.add_argument("--min_batch_timeout", type=int, default=None)
    args = parser.parse_args(db_model)

    inputs = None
    outputs = None

    if args.inputs:
        inputs = list(args.inputs)
    if args.outputs:
        outputs = list(args.outputs)

    name = str(args.name)

    # devices_per_node being greater than one only applies to GPU devices
    if args.devices_per_node > 1 and args.device.lower() == "gpu":
        client.set_model_from_file_multigpu(
            name,
            args.file,
            args.backend,
            0,
            args.devices_per_node,
            args.batch_size,
            args.min_batch_size,
            args.tag,
            inputs,
            outputs,
        )
    else:
        client.set_model_from_file(
            name,
            args.file,
            args.backend,
            args.device,
            args.batch_size,
            args.min_batch_size,
            args.tag,
            inputs,
            outputs,
        )

    return name


def launch_db_script(client: Client, db_script: t.List[str]) -> str:
    """Parse options to launch script on local cluster

    :param client: SmartRedis client connected to local DB
    :type client: Client
    :param db_model: List of arguments defining the script
    :type db_model: List[str]
    :return: Name of model
    :rtype: str
    """
    parser = argparse.ArgumentParser("Set script on DB")
    parser.add_argument("--name", type=str)
    parser.add_argument("--func", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--backend", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--devices_per_node", type=int)
    args = parser.parse_args(db_script)

    if args.file and args.func:
        raise ValueError("Both file and func cannot be provided.")

    if args.func:
        func = args.func.replace("\\n", "\n")
        if args.devices_per_node > 1 and args.device.lower() == "gpu":
            client.set_script_multigpu(args.name, func, 0, args.devices_per_node)
        else:
            client.set_script(args.name, func, args.device)
    elif args.file:
        if args.devices_per_node > 1 and args.device.lower() == "gpu":
            client.set_script_from_file_multigpu(
                args.name, args.file, 0, args.devices_per_node
            )
        else:
            client.set_script_from_file(args.name, args.file, args.device)
    else:
        raise ValueError("No file or func provided.")

    return str(args.name)


def main(
    network_interface: str,
    db_cpus: int,
    command: t.List[str],
    db_models: t.List[t.List[str]],
    db_scripts: t.List[t.List[str]],
) -> None:
    global DBPID  # pylint: disable=global-statement

    lo_address = current_ip("lo")
    try:
        ip_addresses = [
            current_ip(interface) for interface in network_interface.split(",")
        ]

    except ValueError as e:
        logger.warning(e)
        ip_addresses = []

    if all(lo_address == ip_address for ip_address in ip_addresses) or not ip_addresses:
        cmd = command + [f"--bind {lo_address}"]
    else:
        # bind to both addresses if the user specified a network
        # address that exists and is not the loopback address
        cmd = command + [f"--bind {lo_address} {' '.join(ip_addresses)}"]
        # pin source address to avoid random selection by Redis
        cmd += [f"--bind-source-addr {lo_address}"]

    # we generally want to catch all exceptions here as
    # if this process dies, the application will most likely fail
    try:
        process = psutil.Popen(cmd, stdout=PIPE, stderr=STDOUT)
        DBPID = process.pid

    except Exception as e:
        cleanup()
        logger.error(f"Failed to start database process: {str(e)}")
        raise SSInternalError("Colocated process failed to start") from e

    try:
        logger.debug(
            "\n\nColocated database information\n"
            f"\n\tIP Address(es): {' '.join(ip_addresses + [lo_address])}"
            f"\n\tCommand: {' '.join(cmd)}\n\n"
            f"\n\t# of Database CPUs: {db_cpus}"
        )
    except Exception as e:
        cleanup()
        logger.error(f"Failed to start database process: {str(e)}")
        raise SSInternalError("Colocated process failed to start") from e

    def launch_models(client: Client, db_models: t.List[t.List[str]]) -> None:
        for i, db_model in enumerate(db_models):
            logger.debug("Uploading model")
            model_name = launch_db_model(client, db_model)
            logger.debug(f"Added model {model_name} ({i+1}/{len(db_models)})")

    def launch_db_scripts(client: Client, db_scripts: t.List[t.List[str]]) -> None:
        for i, db_script in enumerate(db_scripts):
            logger.debug("Uploading script")
            script_name = launch_db_script(client, db_script)
            logger.debug(f"Added script {script_name} ({i+1}/{len(db_scripts)})")

    try:
        if db_models or db_scripts:
            try:
                client = Client(cluster=False)
                launch_models(client, db_models)
                launch_db_scripts(client, db_scripts)
            except (RedisConnectionError, RedisReplyError) as ex:
                raise SSInternalError(
                    "Failed to set model or script, could not connect to database"
                ) from ex
            finally:
                # Make sure we don't keep this around
                del client

        for line in iter(process.stdout.readline, b""):
            print(line.decode("utf-8").rstrip(), flush=True)

    except Exception as e:
        cleanup()
        logger.error(f"Colocated database process failed: {str(e)}")
        raise SSInternalError("Colocated entrypoint raised an error") from e


def cleanup() -> None:
    try:
        logger.debug("Cleaning up colocated database")
        # attempt to stop the database process
        db_proc = psutil.Process(DBPID)
        db_proc.terminate()

    except psutil.NoSuchProcess:
        logger.warning("Couldn't find database process to kill.")

    except OSError as e:
        logger.warning(f"Failed to clean up colocated database gracefully: {str(e)}")
    finally:
        if LOCK.is_locked:
            LOCK.release()

        if os.path.exists(LOCK.lock_file):
            os.remove(LOCK.lock_file)


def register_signal_handlers() -> None:
    for sig in SIGNALS:
        signal.signal(sig, handle_signal)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prefix_chars="+", description="SmartSim Process Launcher"
    )
    arg_parser.add_argument(
        "+ifname", type=str, help="Network Interface name", default=""
    )
    arg_parser.add_argument(
        "+lockfile", type=str, help="Filename to create for single proc per host"
    )
    arg_parser.add_argument(
        "+db_cpus", type=int, default=2, help="Number of CPUs to use for DB"
    )
    arg_parser.add_argument("+command", nargs="+", help="Command to run")
    arg_parser.add_argument(
        "+db_model",
        nargs="+",
        action="append",
        default=[],
        help="Model to set on DB",
    )
    arg_parser.add_argument(
        "+db_script",
        nargs="+",
        action="append",
        default=[],
        help="Script to set on DB",
    )

    os.environ["PYTHONUNBUFFERED"] = "1"

    try:
        parsed_args = arg_parser.parse_args()
        tmp_lockfile = Path(tempfile.gettempdir()) / parsed_args.lockfile

        LOCK = filelock.FileLock(tmp_lockfile)
        LOCK.acquire(timeout=0.1)
        logger.debug(f"Starting colocated database on host: {socket.gethostname()}")

        # make sure to register the cleanup before the start
        # the proecss so our signaller will be able to stop
        # the database process.
        register_signal_handlers()

        main(
            parsed_args.ifname,
            parsed_args.db_cpus,
            parsed_args.command,
            parsed_args.db_model,
            parsed_args.db_script,
        )

    # gracefully exit the processes in the distributed application that
    # we do not want to have start a colocated process. Only one process
    # per node should be running.
    except filelock.Timeout:
        sys.exit(0)
