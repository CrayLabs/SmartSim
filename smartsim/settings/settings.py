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

import typing as t

from .._core.utils.helpers import is_valid_cmd
from ..error import SmartSimError
from ..settings import (
    AprunSettings,
    BsubBatchSettings,
    Container,
    DragonRunSettings,
    JsrunSettings,
    MpiexecSettings,
    MpirunSettings,
    OrterunSettings,
    PalsMpiexecSettings,
    QsubBatchSettings,
    RunSettings,
    SbatchSettings,
    SrunSettings,
    base,
)
from ..wlm import detect_launcher

_TRunSettingsSelector = t.Callable[[str], t.Callable[..., RunSettings]]


def create_batch_settings(
    launcher: str,
    nodes: t.Optional[int] = None,
    time: str = "",
    queue: t.Optional[str] = None,
    account: t.Optional[str] = None,
    batch_args: t.Optional[t.Dict[str, str]] = None,
    **kwargs: t.Any,
) -> base.BatchSettings:
    """Create a ``BatchSettings`` instance

    See Experiment.create_batch_settings for details

    :param launcher: launcher for this experiment, if set to 'auto',
                     an attempt will be made to find an available launcher on the system
    :param nodes: number of nodes for batch job
    :param time: length of batch job
    :param queue: queue or partition (if slurm)
    :param account: user account name for batch system
    :param batch_args: additional batch arguments
    :return: a newly created BatchSettings instance
    :raises SmartSimError: if batch creation fails
    """
    # all supported batch class implementations
    by_launcher: t.Dict[str, t.Callable[..., base.BatchSettings]] = {
        "pbs": QsubBatchSettings,
        "slurm": SbatchSettings,
        "lsf": BsubBatchSettings,
        "pals": QsubBatchSettings,
    }

    if launcher in ["auto", "dragon"]:
        launcher = detect_launcher()
        if launcher == "dragon":
            by_launcher["dragon"] = by_launcher[launcher]

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
    launcher: str,
    exe: str,
    exe_args: t.Optional[t.List[str]] = None,
    run_command: str = "auto",
    run_args: t.Optional[t.Dict[str, t.Union[int, str, float, None]]] = None,
    env_vars: t.Optional[t.Dict[str, t.Optional[str]]] = None,
    container: t.Optional[Container] = None,
    **kwargs: t.Any,
) -> RunSettings:
    """Create a ``RunSettings`` instance.

    See Experiment.create_run_settings docstring for more details

    :param launcher: launcher to create settings for, if set to 'auto',
                     an attempt will be made to find an available launcher on the system
    :param run_command: command to run the executable
    :param exe: executable to run
    :param exe_args: arguments to pass to the executable
    :param run_args: arguments to pass to the ``run_command``
    :param env_vars: environment variables to pass to the executable
    :param container: container type for workload (e.g. "singularity")
    :return: the created ``RunSettings``
    :raises SmartSimError: if run_command=="auto" and detection fails
    """
    # all supported RunSettings child classes
    supported: t.Dict[str, _TRunSettingsSelector] = {
        "aprun": lambda launcher: AprunSettings,
        "srun": lambda launcher: SrunSettings,
        "mpirun": lambda launcher: MpirunSettings,
        "mpiexec": lambda launcher: (
            MpiexecSettings if launcher != "pals" else PalsMpiexecSettings
        ),
        "orterun": lambda launcher: OrterunSettings,
        "jsrun": lambda launcher: JsrunSettings,
    }

    # run commands supported by each launcher
    # in order of suspected user preference
    by_launcher = {
        "dragon": [""],
        "slurm": ["srun", "mpirun", "mpiexec"],
        "pbs": ["aprun", "mpirun", "mpiexec"],
        "pals": ["mpiexec"],
        "lsf": ["jsrun", "mpirun", "mpiexec"],
        "local": [""],
    }

    if launcher == "auto":
        launcher = detect_launcher()

    def _detect_command(launcher: str) -> str:
        if launcher in by_launcher:
            if launcher in ["local", "dragon"]:
                return ""

            for cmd in by_launcher[launcher]:
                if is_valid_cmd(cmd):
                    return cmd
        msg = (
            "Could not automatically detect a run command to use for launcher "
            f"{launcher}\nSearched for and could not find the following "
            f"commands: {by_launcher[launcher]}"
        )
        raise SmartSimError(msg)

    if run_command:
        run_command = run_command.lower()
    launcher = launcher.lower()

    # detect run_command automatically for all but local launcher
    if run_command == "auto":
        # no auto detection for local, revert to false
        run_command = _detect_command(launcher)

    if launcher == "dragon":
        return DragonRunSettings(
            exe=exe, exe_args=exe_args, env_vars=env_vars, container=container, **kwargs
        )

    # if user specified and supported or auto detection worked
    if run_command and run_command in supported:
        return supported[run_command](launcher)(
            exe, exe_args, run_args, env_vars, container=container, **kwargs
        )

    # 1) user specified and not implementation in SmartSim
    # 2) user supplied run_command=None
    # 3) local launcher being used and default of "auto" was passed.
    return RunSettings(
        exe, exe_args, run_command, run_args, env_vars, container=container
    )
