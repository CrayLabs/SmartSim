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

import dataclasses
import io
import itertools
import random
import re
import time
import typing as t
import uuid
from os import path as osp

import pytest

from smartsim._core import dispatch
from smartsim._core.control.launch_history import LaunchHistory
from smartsim._core.generation.generator import Job_Path
from smartsim._core.utils.launcher import LauncherProtocol, create_job_id
from smartsim.builders.ensemble import Ensemble
from smartsim.entity import entity
from smartsim.entity.application import Application
from smartsim.error import errors
from smartsim.experiment import Experiment
from smartsim.launchable import job
from smartsim.settings import launch_settings
from smartsim.settings.arguments import launch_arguments
from smartsim.status import InvalidJobStatus, JobStatus
from smartsim.types import LaunchedJobID

pytestmark = pytest.mark.group_a

_ID_GENERATOR = (str(i) for i in itertools.count())


def random_id():
    return next(_ID_GENERATOR)


@pytest.fixture
def experiment(monkeypatch, test_dir, dispatcher):
    """A simple experiment instance with a unique name anda unique name and its
    own directory to be used by tests
    """
    exp = Experiment(f"test-exp-{uuid.uuid4()}", test_dir)
    monkeypatch.setattr(dispatch, "DEFAULT_DISPATCHER", dispatcher)
    monkeypatch.setattr(
        exp,
        "_generate",
        lambda generator, job, idx: Job_Path(
            "/tmp/job", "/tmp/job/out.txt", "/tmp/job/err.txt"
        ),
    )
    yield exp


@pytest.fixture
def dispatcher():
    """A pre-configured dispatcher to be used by experiments that simply
    dispatches any jobs with `MockLaunchArgs` to a `NoOpRecordLauncher`
    """
    d = dispatch.Dispatcher()
    to_record: dispatch.FormatterType[MockLaunchArgs, LaunchRecord] = (
        lambda settings, exe, path, env, out, err: LaunchRecord(
            settings, exe, env, path, out, err
        )
    )
    d.dispatch(MockLaunchArgs, with_format=to_record, to_launcher=NoOpRecordLauncher)
    yield d


@pytest.fixture
def job_maker(monkeypatch):
    """A fixture to generate a never ending stream of `Job` instances each
    configured with a unique `MockLaunchArgs` instance, but identical
    executable.
    """

    def iter_jobs():
        for i in itertools.count():
            settings = launch_settings.LaunchSettings("local")
            monkeypatch.setattr(settings, "_arguments", MockLaunchArgs(i))
            yield job.Job(EchoHelloWorldEntity(), settings)

    jobs = iter_jobs()
    yield lambda: next(jobs)


JobMakerType: t.TypeAlias = t.Callable[[], job.Job]


@dataclasses.dataclass(frozen=True, eq=False)
class NoOpRecordLauncher(LauncherProtocol):
    """Simple launcher to track the order of and mapping of ids to `start`
    method calls. It has exactly three attrs:

        - `created_by_experiment`:
              A back ref to the experiment used when calling
              `NoOpRecordLauncher.create`.

        - `launched_order`:
              An append-only list of `LaunchRecord`s that it has "started". Notice
              that this launcher will not actually open any subprocesses/run any
              threads/otherwise execute the contents of the record on the system

        - `ids_to_launched`:
              A mapping where keys are the generated launched id returned from
              a `NoOpRecordLauncher.start` call and the values are the
              `LaunchRecord` that was passed into `NoOpRecordLauncher.start` to
              cause the id to be generated.

    This is helpful for testing that launchers are handling the expected input
    """

    created_by_experiment: Experiment
    launched_order: list[LaunchRecord] = dataclasses.field(default_factory=list)
    ids_to_launched: dict[dispatch.LaunchedJobID, LaunchRecord] = dataclasses.field(
        default_factory=dict
    )

    __hash__ = object.__hash__

    @classmethod
    def create(cls, exp):
        return cls(exp)

    def start(self, record: LaunchRecord):
        id_ = create_job_id()
        self.launched_order.append(record)
        self.ids_to_launched[id_] = record
        return id_

    def get_status(self, *ids):
        raise NotImplementedError

    def stop_jobs(self, *ids):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class LaunchRecord:
    launch_args: launch_arguments.LaunchArguments
    entity: entity.SmartSimEntity
    env: t.Mapping[str, str | None]
    path: str
    out: str
    err: str

    @classmethod
    def from_job(cls, job: job.Job):
        """Create a launch record for what we would expect a launch record to
        look like having gone through the launching process

        :param job: A job that has or will be launched through an experiment
            and dispatched to a `NoOpRecordLauncher`
        :returns: A `LaunchRecord` that should evaluate to being equivilient to
            that of the one stored in the `NoOpRecordLauncher`
        """
        args = job._launch_settings.launch_args
        entity = job._entity.as_executable_sequence()
        env = job._launch_settings.env_vars
        path = "/tmp/job"
        out = "/tmp/job/out.txt"
        err = "/tmp/job/err.txt"
        return cls(args, entity, env, path, out, err)


class MockLaunchArgs(launch_arguments.LaunchArguments):
    """A `LaunchArguments` subclass that will evaluate as true with another if
    and only if they were initialized with the same id. In practice this class
    has no arguments to set.
    """

    def __init__(self, id_: int):
        super().__init__({})
        self.id = id_

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return other.id == self.id

    def launcher_str(self):
        return "test-launch-args"

    def set(self, arg, val): ...


class EchoHelloWorldEntity(entity.SmartSimEntity):
    """A simple smartsim entity"""

    def __init__(self):
        super().__init__("test-entity")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.as_executable_sequence() == other.as_executable_sequence()

    def as_executable_sequence(self):
        return ("echo", "Hello", "World!")


# fmt: off
@pytest.mark.parametrize(
    "num_jobs", [pytest.param(i, id=f"{i} job(s)") for i in (1, 2, 3, 5, 10, 100, 1_000)]
)
@pytest.mark.parametrize(
    "make_jobs", (
        pytest.param(lambda maker, n: tuple(maker() for _ in range(n)), id="many job instances"),
        pytest.param(lambda maker, n: (maker(),) * n                  , id="same job instance many times"),
    ),
)
# fmt: on
def test_start_can_launch_jobs(
    experiment: Experiment,
    job_maker: JobMakerType,
    make_jobs: t.Callable[[JobMakerType, int], tuple[job.Job, ...]],
    num_jobs: int,
) -> None:
    jobs = make_jobs(job_maker, num_jobs)
    assert (
        len(list(experiment._launch_history.iter_past_launchers())) == 0
    ), "Initialized w/ launchers"
    launched_ids = experiment.start(*jobs)
    assert (
        len(list(experiment._launch_history.iter_past_launchers())) == 1
    ), "Unexpected number of launchers"
    ((launcher, exp_cached_ids),) = (
        experiment._launch_history.group_by_launcher().items()
    )
    assert isinstance(launcher, NoOpRecordLauncher), "Unexpected launcher type"
    assert launcher.created_by_experiment is experiment, "Not created by experiment"
    assert (
        len(jobs) == len(launcher.launched_order) == len(launched_ids) == num_jobs
    ), "Inconsistent number of jobs/launched jobs/launched ids/expected number of jobs"
    expected_launched = [LaunchRecord.from_job(job) for job in jobs]

    # Check that `job_a, job_b, job_c, ...` are started in that order when
    # calling `experiemnt.start(job_a, job_b, job_c, ...)`
    assert expected_launched == list(launcher.launched_order), "Unexpected launch order"
    assert sorted(launched_ids) == sorted(exp_cached_ids), "Exp did not cache ids"

    # Similarly, check that `id_a, id_b, id_c, ...` corresponds to
    # `job_a, job_b, job_c, ...` when calling
    # `id_a, id_b, id_c, ... = experiemnt.start(job_a, job_b, job_c, ...)`
    expected_id_map = dict(zip(launched_ids, expected_launched))
    assert expected_id_map == launcher.ids_to_launched, "IDs returned in wrong order"


@pytest.mark.parametrize(
    "num_starts",
    [pytest.param(i, id=f"{i} start(s)") for i in (1, 2, 3, 5, 10, 100, 1_000)],
)
def test_start_can_start_a_job_multiple_times_accross_multiple_calls(
    experiment: Experiment, job_maker: JobMakerType, num_starts: int
) -> None:
    assert (
        len(list(experiment._launch_history.iter_past_launchers())) == 0
    ), "Initialized w/ launchers"
    job = job_maker()
    ids_to_launches = {
        experiment.start(job)[0]: LaunchRecord.from_job(job) for _ in range(num_starts)
    }
    assert (
        len(list(experiment._launch_history.iter_past_launchers())) == 1
    ), "Did not reuse the launcher"
    ((launcher, exp_cached_ids),) = (
        experiment._launch_history.group_by_launcher().items()
    )
    assert isinstance(launcher, NoOpRecordLauncher), "Unexpected launcher type"
    assert len(launcher.launched_order) == num_starts, "Unexpected number launches"

    # Check that a single `job` instance can be launched and re-launched and
    # that `id_a, id_b, id_c, ...` corresponds to
    # `"start_a", "start_b", "start_c", ...` when calling
    # ```py
    # id_a = experiment.start(job)  # "start_a"
    # id_b = experiment.start(job)  # "start_b"
    # id_c = experiment.start(job)  # "start_c"
    # ...
    # ```
    assert ids_to_launches == launcher.ids_to_launched, "Job was not re-launched"
    assert sorted(ids_to_launches) == sorted(exp_cached_ids), "Exp did not cache ids"


class GetStatusLauncher(LauncherProtocol):
    def __init__(self):
        self.id_to_status = {create_job_id(): stat for stat in JobStatus}

    __hash__ = object.__hash__

    @property
    def known_ids(self):
        return tuple(self.id_to_status)

    @classmethod
    def create(cls, _):
        raise NotImplementedError("{type(self).__name__} should not be created")

    def start(self, _):
        raise NotImplementedError("{type(self).__name__} should not start anything")

    def _assert_ids(self, ids: LaunchedJobID):
        if any(id_ not in self.id_to_status for id_ in ids):
            raise errors.LauncherJobNotFound

    def get_status(self, *ids: LaunchedJobID):
        self._assert_ids(ids)
        return {id_: self.id_to_status[id_] for id_ in ids}

    def stop_jobs(self, *ids: LaunchedJobID):
        self._assert_ids(ids)
        stopped = {id_: JobStatus.CANCELLED for id_ in ids}
        self.id_to_status |= stopped
        return stopped


@pytest.fixture
def make_populated_experiment(monkeypatch, experiment):
    def impl(num_active_launchers):
        new_launchers = (GetStatusLauncher() for _ in range(num_active_launchers))
        id_to_launcher = {
            id_: launcher for launcher in new_launchers for id_ in launcher.known_ids
        }
        monkeypatch.setattr(
            experiment, "_launch_history", LaunchHistory(id_to_launcher)
        )
        return experiment

    yield impl


def test_experiment_can_get_statuses(make_populated_experiment):
    exp = make_populated_experiment(num_active_launchers=1)
    (launcher,) = exp._launch_history.iter_past_launchers()
    ids = tuple(launcher.known_ids)
    recieved_stats = exp.get_status(*ids)
    assert len(recieved_stats) == len(ids), "Unexpected number of statuses"
    assert (
        dict(zip(ids, recieved_stats)) == launcher.id_to_status
    ), "Statuses in wrong order"


@pytest.mark.parametrize(
    "num_launchers",
    [pytest.param(i, id=f"{i} launcher(s)") for i in (2, 3, 5, 10, 20, 100)],
)
def test_experiment_can_get_statuses_from_many_launchers(
    make_populated_experiment, num_launchers
):
    exp = make_populated_experiment(num_active_launchers=num_launchers)
    launcher_and_rand_ids = (
        (launcher, random.choice(tuple(launcher.id_to_status)))
        for launcher in exp._launch_history.iter_past_launchers()
    )
    expected_id_to_stat = {
        id_: launcher.id_to_status[id_] for launcher, id_ in launcher_and_rand_ids
    }
    query_ids = tuple(expected_id_to_stat)
    stats = exp.get_status(*query_ids)
    assert len(stats) == len(expected_id_to_stat), "Unexpected number of statuses"
    assert dict(zip(query_ids, stats)) == expected_id_to_stat, "Statuses in wrong order"


def test_get_status_returns_not_started_for_unrecognized_ids(
    monkeypatch, make_populated_experiment
):
    exp = make_populated_experiment(num_active_launchers=1)
    brand_new_id = create_job_id()
    ((launcher, (id_not_known_by_exp, *rest)),) = (
        exp._launch_history.group_by_launcher().items()
    )
    new_history = LaunchHistory({id_: launcher for id_ in rest})
    monkeypatch.setattr(exp, "_launch_history", new_history)
    expected_stats = (InvalidJobStatus.NEVER_STARTED,) * 2
    actual_stats = exp.get_status(brand_new_id, id_not_known_by_exp)
    assert expected_stats == actual_stats


def test_get_status_de_dups_ids_passed_to_launchers(
    monkeypatch, make_populated_experiment
):
    def track_calls(fn):
        calls = []

        def impl(*a, **kw):
            calls.append((a, kw))
            return fn(*a, **kw)

        return calls, impl

    exp = make_populated_experiment(num_active_launchers=1)
    ((launcher, (id_, *_)),) = exp._launch_history.group_by_launcher().items()
    calls, tracked_get_status = track_calls(launcher.get_status)
    monkeypatch.setattr(launcher, "get_status", tracked_get_status)
    stats = exp.get_status(id_, id_, id_)
    assert len(stats) == 3, "Unexpected number of statuses"
    assert all(stat == stats[0] for stat in stats), "Statuses are not eq"
    assert len(calls) == 1, "Launcher's `get_status` was called more than once"
    (call,) = calls
    assert call == ((id_,), {}), "IDs were not de-duplicated"


def test_wait_handles_empty_call_args(experiment):
    """An exception is raised when there are no jobs to complete"""
    with pytest.raises(ValueError, match="No job ids"):
        experiment.wait()


def test_wait_does_not_block_unknown_id(experiment):
    """If an experiment does not recognize a job id, it should not wait for its
    completion
    """
    now = time.perf_counter()
    experiment.wait(create_job_id())
    assert time.perf_counter() - now < 1


def test_wait_calls_prefered_impl(make_populated_experiment, monkeypatch):
    """Make wait is calling the expected method for checking job statuses.
    Right now we only have the "polling" impl, but in future this might change
    to an event based system.
    """
    exp = make_populated_experiment(1)
    ((_, (id_, *_)),) = exp._launch_history.group_by_launcher().items()
    was_called = False

    def mocked_impl(*args, **kwargs):
        nonlocal was_called
        was_called = True

    monkeypatch.setattr(exp, "_poll_for_statuses", mocked_impl)
    exp.wait(id_)
    assert was_called


@pytest.mark.parametrize(
    "num_polls",
    [
        pytest.param(i, id=f"Poll for status {i} times")
        for i in (1, 5, 10, 20, 100, 1_000)
    ],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_poll_status_blocks_until_job_is_completed(
    monkeypatch, make_populated_experiment, num_polls, verbose
):
    """Make sure that the polling based implementation blocks the calling
    thread. Use varying number of polls to simulate varying lengths of job time
    for a job to complete.

    Additionally check to make sure that the expected log messages are present
    """
    exp = make_populated_experiment(1)
    ((launcher, (id_, *_)),) = exp._launch_history.group_by_launcher().items()
    (current_status,) = launcher.get_status(id_).values()
    different_statuses = set(JobStatus) - {current_status}
    (new_status, *_) = different_statuses
    mock_log = io.StringIO()

    @dataclasses.dataclass
    class ChangeStatusAfterNPolls:
        n: int
        from_: JobStatus
        to: JobStatus
        num_calls: int = dataclasses.field(default=0, init=False)

        def __call__(self, *args, **kwargs):
            self.num_calls += 1
            ret_status = self.to if self.num_calls >= self.n else self.from_
            return (ret_status,)

    mock_get_status = ChangeStatusAfterNPolls(num_polls, current_status, new_status)
    monkeypatch.setattr(exp, "get_status", mock_get_status)
    monkeypatch.setattr(
        "smartsim.experiment.logger.info", lambda s: mock_log.write(f"{s}\n")
    )
    final_statuses = exp._poll_for_statuses(
        [id_], different_statuses, timeout=10, interval=0, verbose=verbose
    )
    assert final_statuses == {id_: new_status}

    expected_log = io.StringIO()
    expected_log.writelines(
        f"Job({id_}): Running with status '{current_status.value}'\n"
        for _ in range(num_polls - 1)
    )
    expected_log.write(f"Job({id_}): Finished with status '{new_status.value}'\n")
    assert mock_get_status.num_calls == num_polls
    assert mock_log.getvalue() == (expected_log.getvalue() if verbose else "")


def test_poll_status_raises_when_called_with_infinite_iter_wait(
    make_populated_experiment,
):
    """Cannot wait forever between polls. That will just block the thread after
    the first poll
    """
    exp = make_populated_experiment(1)
    ((_, (id_, *_)),) = exp._launch_history.group_by_launcher().items()
    with pytest.raises(ValueError, match="Polling interval cannot be infinite"):
        exp._poll_for_statuses(
            [id_],
            [],
            timeout=10,
            interval=float("inf"),
        )


def test_poll_for_status_raises_if_ids_not_found_within_timeout(
    make_populated_experiment,
):
    """If there is a timeout, a timeout error should be raised when it is exceeded"""
    exp = make_populated_experiment(1)
    ((launcher, (id_, *_)),) = exp._launch_history.group_by_launcher().items()
    (current_status,) = launcher.get_status(id_).values()
    different_statuses = set(JobStatus) - {current_status}
    with pytest.raises(
        TimeoutError,
        match=re.escape(
            f"Job ID(s) {id_} failed to reach terminal status before timeout"
        ),
    ):
        exp._poll_for_statuses(
            [id_],
            different_statuses,
            timeout=1,
            interval=0,
        )


@pytest.mark.parametrize(
    "num_launchers",
    [pytest.param(i, id=f"{i} launcher(s)") for i in (2, 3, 5, 10, 20, 100)],
)
@pytest.mark.parametrize(
    "select_ids",
    [
        pytest.param(
            lambda history: history._id_to_issuer.keys(), id="All launched jobs"
        ),
        pytest.param(
            lambda history: next(iter(history.group_by_launcher().values())),
            id="All from one launcher",
        ),
        pytest.param(
            lambda history: itertools.chain.from_iterable(
                random.sample(tuple(ids), len(JobStatus) // 2)
                for ids in history.group_by_launcher().values()
            ),
            id="Subset per launcher",
        ),
        pytest.param(
            lambda history: random.sample(
                tuple(history._id_to_issuer), len(history._id_to_issuer) // 3
            ),
            id=f"Random subset across all launchers",
        ),
    ],
)
def test_experiment_can_stop_jobs(make_populated_experiment, num_launchers, select_ids):
    exp = make_populated_experiment(num_launchers)
    ids = (launcher.known_ids for launcher in exp._launch_history.iter_past_launchers())
    ids = tuple(itertools.chain.from_iterable(ids))
    before_stop_stats = exp.get_status(*ids)
    to_cancel = tuple(select_ids(exp._launch_history))
    stats = exp.stop(*to_cancel)
    after_stop_stats = exp.get_status(*ids)
    assert stats == (JobStatus.CANCELLED,) * len(to_cancel)
    assert dict(zip(ids, before_stop_stats)) | dict(zip(to_cancel, stats)) == dict(
        zip(ids, after_stop_stats)
    )


def test_experiment_raises_if_asked_to_stop_no_jobs(experiment):
    with pytest.raises(ValueError, match="No job ids provided"):
        experiment.stop()


@pytest.mark.parametrize(
    "num_launchers",
    [pytest.param(i, id=f"{i} launcher(s)") for i in (2, 3, 5, 10, 20, 100)],
)
def test_experiment_stop_does_not_raise_on_unknown_job_id(
    make_populated_experiment, num_launchers
):
    exp = make_populated_experiment(num_launchers)
    new_id = create_job_id()
    all_known_ids = tuple(exp._launch_history._id_to_issuer)
    before_cancel = exp.get_status(*all_known_ids)
    (stat,) = exp.stop(new_id)
    assert stat == InvalidJobStatus.NEVER_STARTED
    after_cancel = exp.get_status(*all_known_ids)
    assert before_cancel == after_cancel


def test_start_raises_if_no_args_supplied(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(ValueError, match="No jobs provided to start"):
        exp.start()


def test_stop_raises_if_no_args_supplied(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(ValueError, match="No job ids provided"):
        exp.stop()


def test_get_status_raises_if_no_args_supplied(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(ValueError, match="No job ids provided"):
        exp.get_status()


def test_poll_raises_if_no_args_supplied(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(
        TypeError, match="missing 2 required positional arguments: 'ids' and 'statuses'"
    ):
        exp._poll_for_statuses()


def test_wait_raises_if_no_args_supplied(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(ValueError, match="No job ids to wait on provided"):
        exp.wait()


def test_type_experiment_name_parameter(test_dir):
    with pytest.raises(TypeError, match="name argument was not of type str"):
        Experiment(name=1, exp_path=test_dir)


def test_type_start_parameters(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(TypeError, match="jobs argument was not of type Job"):
        exp.start("invalid")


def test_type_get_status_parameters(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(TypeError, match="ids argument was not of type LaunchedJobID"):
        exp.get_status(2)


def test_type_wait_parameter(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(TypeError, match="ids argument was not of type LaunchedJobID"):
        exp.wait(2)


def test_type_stop_parameter(test_dir):
    exp = Experiment(name="exp_name", exp_path=test_dir)
    with pytest.raises(TypeError, match="ids argument was not of type LaunchedJobID"):
        exp.stop(2)


@pytest.mark.parametrize(
    "job_list",
    (
        pytest.param(
            [
                (
                    job.Job(
                        Application(
                            "test_name",
                            exe="echo",
                            exe_args=["spam", "eggs"],
                        ),
                        launch_settings.LaunchSettings("local"),
                    ),
                    Ensemble("ensemble-name", "echo", replicas=2).build_jobs(
                        launch_settings.LaunchSettings("local")
                    ),
                )
            ],
            id="(job1, (job2, job_3))",
        ),
        pytest.param(
            [
                (
                    Ensemble("ensemble-name", "echo", replicas=2).build_jobs(
                        launch_settings.LaunchSettings("local")
                    ),
                    (
                        job.Job(
                            Application(
                                "test_name",
                                exe="echo",
                                exe_args=["spam", "eggs"],
                            ),
                            launch_settings.LaunchSettings("local"),
                        ),
                        job.Job(
                            Application(
                                "test_name_2",
                                exe="echo",
                                exe_args=["spam", "eggs"],
                            ),
                            launch_settings.LaunchSettings("local"),
                        ),
                    ),
                )
            ],
            id="((job1, job2), (job3, job4))",
        ),
        pytest.param(
            [
                (
                    job.Job(
                        Application(
                            "test_name",
                            exe="echo",
                            exe_args=["spam", "eggs"],
                        ),
                        launch_settings.LaunchSettings("local"),
                    ),
                )
            ],
            id="(job,)",
        ),
        pytest.param(
            [
                [
                    job.Job(
                        Application(
                            "test_name",
                            exe="echo",
                            exe_args=["spam", "eggs"],
                        ),
                        launch_settings.LaunchSettings("local"),
                    ),
                    (
                        Ensemble("ensemble-name", "echo", replicas=2).build_jobs(
                            launch_settings.LaunchSettings("local")
                        ),
                        job.Job(
                            Application(
                                "test_name_2",
                                exe="echo",
                                exe_args=["spam", "eggs"],
                            ),
                            launch_settings.LaunchSettings("local"),
                        ),
                    ),
                ]
            ],
            id="[job_1, ((job_2, job_3), job_4)]",
        ),
    ),
)
def test_start_unpack(
    test_dir: str, wlmutils, monkeypatch: pytest.MonkeyPatch, job_list: job.Job
):
    """Test unpacking a sequences of jobs"""

    monkeypatch.setattr(
        "smartsim._core.dispatch._LauncherAdapter.start",
        lambda launch, exe, job_execution_path, env, out, err: random_id(),
    )

    exp = Experiment(name="exp_name", exp_path=test_dir)
    exp.start(*job_list)
