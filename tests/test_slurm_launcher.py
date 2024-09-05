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


from __future__ import annotations

import itertools
import os
import pathlib
import shutil
import time
import typing as t

import pytest

from smartsim._core.launcher_.slurm.slurm_launcher import (
    SlurmLauncher,
    SrunCommand,
    _LaunchedJobInfo,
)
from smartsim._core.utils.launcher import create_job_id
from smartsim.error import errors
from smartsim.status import JobStatus

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_srun_command_raises_if_no_alloc_provided(monkeypatch, test_dir):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(errors.AllocationError):
        SrunCommand(
            "HelloWorld",
            ["-N", "1", "-n", "3", f"--chdir={os.fspath(test_dir)}"],
            ["echo", "hello", "world"],
            None,
            {},
        )


def test_srun_command_appends_job_tracking_flags(monkeypatch):
    monkeypatch.setattr(
        "smartsim._core.utils.helpers.expand_exe_path",
        lambda exe: f"/full/path/to/{exe}",
    )
    monkeypatch.setattr(
        "smartsim._core.utils.helpers.create_short_id_str", lambda: "12345"
    )
    srun = SrunCommand(
        "HelloWorld",
        ["-N", "1", "-n", "3"],
        ["echo", "hello", "world"],
        "mock-job-id",
        {},
    )
    assert srun.as_command_line_args() == (
        "/full/path/to/srun",
        "-N",
        "1",
        "-n",
        "3",
        "--job-name=HelloWorld-12345",
        "--jobid=mock-job-id",
        "--",
        "echo",
        "hello",
        "world",
    )


@pytest.fixture
def make_srun_command(test_dir):
    def inner(
        srun_flags: t.Sequence[str],
        exe: t.Sequence[str],
        *,
        use_current_alloc: bool = False,
    ) -> SrunCommand:
        *_, name = exe[0].split(os.path.sep)
        return SrunCommand(
            name,
            [
                *srun_flags,
                f"--chdir={os.fspath(test_dir)}",
                f"--output={os.path.join(test_dir, f'{name}.out')}",
                f"--error={os.path.join(test_dir, f'{name}.err')}",
            ],
            exe,
            job_id=None if use_current_alloc else "MOCK-JOB-ID",
        )

    yield inner


def test_slurm_launcher_can_start_a_command(monkeypatch, make_srun_command):
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_commands.sacct",
        lambda *_, **__: ("out", "err"),
    )
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_parser.parse_step_id_from_sacct",
        lambda *_, **__: "mock-step-id",
    )
    launcher = SlurmLauncher()
    srun = make_srun_command(["-N", "1", "-n", "1"], ["echo", "spam", "eggs"])
    id_ = launcher.start(srun)
    info = launcher._launched[id_]
    assert info.slurm_id == "mock-step-id"
    assert info.name == srun.name
    assert info.status_override is None


def test_slurm_launcher_errors_if_cannot_parse_id(monkeypatch, make_srun_command):
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_commands.sacct",
        lambda *_, **__: ("out", "err"),
    )
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_parser.parse_step_id_from_sacct",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(time, "sleep", lambda *_, **__: ...)
    launcher = SlurmLauncher()
    srun = make_srun_command(["-N", "1", "-n", "1"], ["echo", "spam", "eggs"])
    with pytest.raises(
        errors.LauncherError, match=r"Could not find id of launched job step"
    ):
        launcher.start(srun)


def fail_if_called(*_, **__):
    assert False, "Function unexpectedly called"


def test_slurm_launcher_will_not_fetch_statuses_if_any_id_is_not_recognized(
    monkeypatch,
):
    known_id = create_job_id()
    launcher = SlurmLauncher(
        launched={known_id: _LaunchedJobInfo("mock-id", "mock-job")}
    )
    unknown_id = create_job_id()

    monkeypatch.setattr(launcher, "_get_status", fail_if_called)
    with pytest.raises(
        errors.LauncherError, match=f"has not launched a job with id `{unknown_id}`"
    ):
        launcher.get_status(known_id, unknown_id)


@pytest.mark.parametrize(
    "launched",
    (
        pytest.param(
            {
                create_job_id(): _LaunchedJobInfo("mock-id-1", "mock-job-1"),
                create_job_id(): _LaunchedJobInfo("mock-id-2", "mock-job-2"),
            },
            id="No override",
        ),
        pytest.param(
            {
                create_job_id(): _LaunchedJobInfo(
                    "mock-id-1", "mock-job-1", status_override=JobStatus.FAILED
                ),
                create_job_id(): _LaunchedJobInfo(
                    "mock-id-2", "mock-job-2", status_override=JobStatus.CANCELLED
                ),
            },
            id="override",
        ),
        pytest.param(
            {
                create_job_id(): _LaunchedJobInfo("mock-id-1", "mock-job-1"),
                create_job_id(): _LaunchedJobInfo(
                    "mock-id-2", "mock-job-2", status_override=JobStatus.CANCELLED
                ),
                create_job_id(): _LaunchedJobInfo("mock-id-3", "mock-job-3"),
            },
            id="Both overrids and no override",
        ),
    ),
)
@pytest.mark.parametrize(
    "mock_sacct_out, mock_fetch_status",
    (
        pytest.param("RUNNING", JobStatus.RUNNING, id="running"),
        pytest.param("CONFIGURING", JobStatus.RUNNING, id="configuring"),
        pytest.param("STAGE_OUT", JobStatus.RUNNING, id="stage_out"),
        pytest.param("COMPLETED", JobStatus.COMPLETED, id="completed"),
        pytest.param("DEADLINE", JobStatus.COMPLETED, id="deadline"),
        pytest.param("TIMEOUT", JobStatus.COMPLETED, id="timeout"),
        pytest.param("BOOT_FAIL", JobStatus.FAILED, id="boot_fail"),
        pytest.param("FAILED", JobStatus.FAILED, id="failed"),
        pytest.param("NODE_FAIL", JobStatus.FAILED, id="node_fail"),
        pytest.param("OUT_OF_MEMORY", JobStatus.FAILED, id="out_of_memory"),
        pytest.param("CANCELLED", JobStatus.CANCELLED, id="cancelled"),
        pytest.param("CANCELLED+", JobStatus.CANCELLED, id="cancelled"),
        pytest.param("REVOKED", JobStatus.CANCELLED, id="revoked"),
        pytest.param("PENDING", JobStatus.PAUSED, id="pending"),
        pytest.param("PREEMPTED", JobStatus.PAUSED, id="preempted"),
        pytest.param("RESV_DEL_HOLD", JobStatus.PAUSED, id="resv_del_hold"),
        pytest.param("REQUEUE_FED", JobStatus.PAUSED, id="requeue_fed"),
        pytest.param("REQUEUE_HOLD", JobStatus.PAUSED, id="requeue_hold"),
        pytest.param("REQUEUED", JobStatus.PAUSED, id="requeued"),
        pytest.param("RESIZING", JobStatus.PAUSED, id="resizing"),
        pytest.param("SIGNALING", JobStatus.PAUSED, id="signaling"),
        pytest.param("SPECIAL_EXIT", JobStatus.PAUSED, id="special_exit"),
        pytest.param("STOPPED", JobStatus.PAUSED, id="stopped"),
        pytest.param("SUSPENDED", JobStatus.PAUSED, id="suspended"),
        pytest.param("NONSENSE", JobStatus.UNKNOWN, id="nonsense"),
    ),
)
def test_slurm_will_fetch_statuses(
    monkeypatch, launched, mock_sacct_out, mock_fetch_status
):
    launcher = SlurmLauncher(launched=launched)
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_commands.sacct",
        lambda *_, **__: ("out", "err"),
    )
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_parser.parse_sacct",
        lambda *_, **__: (mock_sacct_out,),
    )
    assert launcher.get_status(*launched) == {
        k: v.status_override or mock_fetch_status for k, v in launched.items()
    }


def test_slurm_launcher_will_not_stop_jobs_if_any_id_is_not_recognized(monkeypatch):
    known_id = create_job_id()
    launcher = SlurmLauncher(
        launched={known_id: _LaunchedJobInfo("mock-id", "mock-job")}
    )
    unknown_id = create_job_id()

    monkeypatch.setattr(launcher, "_stop_job", fail_if_called)
    with pytest.raises(
        errors.LauncherError, match=f"has not launched a job with id `{unknown_id}`"
    ):
        launcher.get_status(known_id, unknown_id)


@pytest.mark.parametrize("is_het_job", [False, True])
@pytest.mark.parametrize("scancel_rc", [0, 123])
def test_slurm_launcher_stops_jobs(monkeypatch, is_het_job, scancel_rc):
    if is_het_job:
        monkeypatch.setenv("SLURM_HET_SIZE", "123456")
    else:
        monkeypatch.delenv("SLURM_HET_SIZE", raising=False)
    monkeypatch.setattr(
        "smartsim._core.launcher_.slurm.slurm_commands.scancel",
        lambda *_, **__: (scancel_rc, "out", "err"),
    )
    id_ = create_job_id()
    info = _LaunchedJobInfo("mock-id", "mock-job")
    launcher = SlurmLauncher(launched={id_: info})
    monkeypatch.setattr(
        launcher,
        "get_status",
        lambda *ids: dict(zip(ids, itertools.repeat(JobStatus.CANCELLED))),
    )
    assert launcher.stop_jobs(id_) == {id_: JobStatus.CANCELLED}
    assert info.status_override == (
        JobStatus.CANCELLED if is_het_job and scancel_rc != 0 else None
    )


requires_slurm = pytest.mark.skipif(
    shutil.which("srun") is None
    or shutil.which("sbatch") is None
    or shutil.which("sacct") is None
    or shutil.which("scancel") is None,
    reason="Slurm utilities could be found",
)


def requires_alloc_size(num_nodes):
    try:
        alloc_size = int(os.environ.get("SLURM_NNODES", None))
    except (TypeError, ValueError):
        alloc_size = None
    return pytest.mark.skipif(
        alloc_size is None or alloc_size < num_nodes,
        reason=f"Test requires an allocation with at least {num_nodes} nodes",
    )


@requires_slurm
@requires_alloc_size(1)
def test_srun_hello_world(make_srun_command, test_dir):
    launcher = SlurmLauncher()
    srun = make_srun_command(
        ["-N", "1", "-n", "3"], ["echo", "hello world"], use_current_alloc=True
    )
    id_ = launcher.start(srun)
    time.sleep(1)
    assert launcher.get_status(id_)[id_] == JobStatus.COMPLETED
    with open(os.path.join(test_dir, "echo.out"), "r") as fd:
        assert fd.read() == "hello world\n" * 3


@pytest.mark.xfail(reason=r"Slurm launcher cannout parse `CANCELLED by \d+` syntax")
@requires_slurm
@requires_alloc_size(1)
def test_srun_sleep_for_two_min_with_cancel(make_srun_command):
    launcher = SlurmLauncher()
    srun = make_srun_command(
        ["-N", "1", "-n", "1"], ["sleep", "120"], use_current_alloc=True
    )
    id_ = launcher.start(srun)
    time.sleep(1)
    assert launcher.get_status(id_)[id_] == JobStatus.RUNNING
    launcher.stop_jobs(id_)
    time.sleep(1)
    assert launcher.get_status(id_)[id_] == JobStatus.CANCELLED
