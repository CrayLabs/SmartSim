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
# import asyncio
import json
import logging
import os
import pathlib
import typing as t

from smartsim._core.launcher.stepInfo import StepInfo
from smartsim.status import TERMINAL_STATUSES, SmartSimStatus

_EventClass = t.Literal["start", "stop", "timestep"]

logger = logging.getLogger("TelemetryMonitor")


def write_event(
    timestamp: int,
    task_id: t.Union[int, str],
    step_id: str,
    etype: str,
    event_type: _EventClass,
    status_dir: pathlib.Path,
    detail: str = "",
    return_code: t.Optional[int] = None,
) -> None:
    """Write a record to durable storage for a SmartSimEntity lifecycle event

    :param timestamp: when the event occurred
    :type timestamp: str
    :param task_id: the task_id of a managed task
    :type task_id: int|str
    :param step_id: the step_id of an unmanaged task
    :type step_id: str
    :param etype: the SmartSimEntity subtype (e.g. `orchestrator`, `ensemble`, ...)
    :type etype: str
    :param event_type: the event subtype
    :type event_type: _EventClass
    :param status_dir: (optional) path where the SmartSimEntity outputs are written
    :type status_dir: pathlib.Path
    :param detail: (optional) additional information to write with the event
    :type detail: str
    :param return_code: (optional) the return code of a completed task
    :type return_code: str|None"""
    tgt_path = status_dir / f"{event_type}.json"
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        task_id = int(task_id)
    except ValueError:
        pass

    entity_dict = {
        "timestamp": timestamp,
        "job_id": task_id,
        "step_id": step_id,
        "type": etype,
        "action": event_type,
    }

    if detail is not None:
        entity_dict["detail"] = detail

    if return_code is not None:
        entity_dict["return_code"] = return_code

    try:
        if not tgt_path.exists():
            # Don't overwrite existing tracking files
            bytes_written = tgt_path.write_text(json.dumps(entity_dict, indent=2))
            if bytes_written < 1:
                logger.warning("event tracking failed to write tracking file.")
    except Exception:
        logger.error("Unable to write tracking file.", exc_info=True)


def map_return_code(step_info: StepInfo) -> t.Optional[int]:
    """Converts a return code from a workload manager into a SmartSim status.

    A non-terminal status is converted to null. This indicates
    that the process referenced in the `StepInfo` is running
    and does not yet have a return code.

    :param step_info: (optional) the return code of a completed task
    :type step_info: StepInfo

    :return: a return code if the step is finished, otherwise None
    :rtype: int"""
    rc_map = {s: 1 for s in TERMINAL_STATUSES}  # return `1` for all terminal statuses
    rc_map.update(
        {SmartSimStatus.STATUS_COMPLETED: os.EX_OK}
    )  # return `0` for full success

    return rc_map.get(step_info.status, None)  # return `None` when in-progress
