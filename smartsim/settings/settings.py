# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from .._core.utils.helpers import is_valid_cmd
from ..error import SmartSimError
from ..wlm import detect_launcher
from . import *


def create_batch_settings(
    launcher, nodes=None, time="", queue=None, account=None, batch_args=None, **kwargs
):
    """Create a ``BatchSettings`` instance

    See Experiment.create_batch_settings for details

    :param launcher: launcher for this experiment, if set to 'auto',
                     an attempt will be made to find an available launcher on the system
    :type launcher: str
    :param nodes: number of nodes for batch job, defaults to 1
    :type nodes: int, optional
    :param time: length of batch job, defaults to ""
    :type time: str, optional
    :param queue: queue or partition (if slurm), defaults to ""
    :type queue: str, optional
    :param account: user account name for batch system, defaults to ""
    :type account: str, optional
    :param batch_args: additional batch arguments, defaults to None
    :type batch_args: dict[str, str], optional
    :return: a newly created BatchSettings instance
    :rtype: BatchSettings
    :raises SmartSimError: if batch creation fails
    """
    # all supported batch class implementations
    by_launcher = {
        "cobalt": CobaltBatchSettings,
        "pbs": QsubBatchSettings,
        "slurm": SbatchSettings,
        "lsf": BsubBatchSettings,
    }

    if launcher == "auto":
        launcher = detect_launcher()

    if launcher == "local":
        raise SmartSimError("Local launcher does not support batch workloads")

    # detect the batch class to use based on the launcher provided by
    # the user
    try:
        batch_class = by_launcher[launcher]
        batch_settings = batch_class(
            nodes=nodes,
            time=time,
            batch_args=batch_args,
            queue=queue,
            account=account,
            **kwargs,
        )
        return batch_settings

    except KeyError:
        raise SmartSimError(
            f"User attempted to make batch settings for unsupported launcher {launcher}"
        ) from None


def create_run_settings(
    launcher,
    exe,
    exe_args=None,
    run_command="auto",
    run_args=None,
    env_vars=None,
    **kwargs,
):
    """Create a ``RunSettings`` instance.

    See Experiment.create_run_settings docstring for more details

    :param launcher: launcher to create settings for, if set to 'auto',
                     an attempt will be made to find an available launcher on the system
    :type launcher: str
    :param run_command: command to run the executable
    :type run_command: str
    :param exe: executable to run
    :type exe: str
    :param exe_args: arguments to pass to the executable
    :type exe_args: list[str], optional
    :param run_args: arguments to pass to the ``run_command``
    :type run_args: list[str], optional
    :param env_vars: environment variables to pass to the executable
    :type env_vars: dict[str, str], optional
    :return: the created ``RunSettings``
    :rtype: RunSettings
    :raises SmartSimError: if run_command=="auto" and detection fails
    """
    # all supported RunSettings child classes
    supported = {
        "aprun": AprunSettings,
        "srun": SrunSettings,
        "mpirun": MpirunSettings,
        "jsrun": JsrunSettings,
    }

    # run commands supported by each launcher
    # in order of suspected user preference
    by_launcher = {
        "slurm": ["srun", "mpirun"],
        "pbs": ["aprun", "mpirun"],
        "cobalt": ["aprun", "mpirun"],
        "lsf": ["jsrun", "mpirun"],
    }

    if launcher == "auto":
        launcher = detect_launcher()

    def _detect_command(launcher):
        if launcher in by_launcher:
            for cmd in by_launcher[launcher]:
                if is_valid_cmd(cmd):
                    return cmd
        msg = f"Could not automatically detect a run command to use for launcher {launcher}"
        msg += f"\nSearched for and could not find the following commands: {by_launcher[launcher]}"
        raise SmartSimError(msg)

    if run_command:
        run_command = run_command.lower()
    launcher = launcher.lower()

    # detect run_command automatically for all but local launcher
    if run_command == "auto":
        # no auto detection for local, revert to false
        if launcher == "local":
            run_command = None
        else:
            run_command = _detect_command(launcher)

    # if user specified and supported or auto detection worked
    if run_command and run_command in supported:
        return supported[run_command](exe, exe_args, run_args, env_vars, **kwargs)

    # 1) user specified and not implementation in SmartSim
    # 2) user supplied run_command=None
    # 3) local launcher being used and default of "auto" was passed.
    return RunSettings(exe, exe_args, run_command, run_args, env_vars)
