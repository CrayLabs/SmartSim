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
import itertools
import random
import tempfile
import typing as t
import uuid

import pytest

from smartsim._core import dispatch
from smartsim._core.control.launch_history import LaunchHistory
from smartsim._core.utils.launcher import LauncherProtocol, create_job_id
from smartsim.entity import _mock, entity
from smartsim.experiment import Experiment
from smartsim.launchable import job
from smartsim.settings import launchSettings
from smartsim.settings.arguments import launchArguments
from smartsim.status import InvalidJobStatus, JobStatus

pytestmark = pytest.mark.group_a


@pytest.fixture
def experiment(monkeypatch, test_dir, dispatcher):
    """A simple experiment instance with a unique name anda unique name and its
    own directory to be used by tests
    """
    exp = Experiment(f"test-exp-{uuid.uuid4()}", test_dir)
    monkeypatch.setattr(dispatch, "DEFAULT_DISPATCHER", dispatcher)
    monkeypatch.setattr(exp, "_generate", lambda gen, job, idx: "/tmp/job")
    yield exp


@pytest.fixture
def dispatcher():
    """A pre-configured dispatcher to be used by experiments that simply
    dispatches any jobs with `MockLaunchArgs` to a `NoOpRecordLauncher`
    """
    d = dispatch.Dispatcher()
    to_record: dispatch._FormatterType[MockLaunchArgs, LaunchRecord] = (
        lambda settings, exe, path, env: LaunchRecord(settings, exe, env, path)
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
            settings = launchSettings.LaunchSettings("local")
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


@dataclasses.dataclass(frozen=True)
class LaunchRecord:
    launch_args: launchArguments.LaunchArguments
    entity: entity.SmartSimEntity
    env: t.Mapping[str, str | None]
    path: str

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
        entity = job._entity
        env = job._launch_settings.env_vars
        path = "/tmp/job"
        return cls(args, entity, env, path)


class MockLaunchArgs(launchArguments.LaunchArguments):
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
    """A simple smartsim entity that meets the `ExecutableProtocol` protocol"""

    def __init__(self):
        super().__init__("test-entity")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.as_program_arguments() == other.as_program_arguments()

    def as_program_arguments(self):
        return ("echo", "Hello", "World!")


def test_start_raises_if_no_args_supplied(experiment):
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        experiment.start()


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

    def get_status(self, *ids: LaunchedJobID):
        return {id_: self.id_to_status[id_] for id_ in ids}


@pytest.fixture
def make_populated_experment(monkeypatch, experiment):
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


def test_experiment_can_get_statuses(make_populated_experment):
    exp = make_populated_experment(num_active_launchers=1)
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
    make_populated_experment, num_launchers
):
    exp = make_populated_experment(num_active_launchers=num_launchers)
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
    monkeypatch, make_populated_experment
):
    exp = make_populated_experment(num_active_launchers=1)
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
    monkeypatch, make_populated_experment
):
    def track_calls(fn):
        calls = []

        def impl(*a, **kw):
            calls.append((a, kw))
            return fn(*a, **kw)

        return calls, impl

    exp = make_populated_experment(num_active_launchers=1)
    ((launcher, (id_, *_)),) = exp._launch_history.group_by_launcher().items()
    calls, tracked_get_status = track_calls(launcher.get_status)
    monkeypatch.setattr(launcher, "get_status", tracked_get_status)
    stats = exp.get_status(id_, id_, id_)
    assert len(stats) == 3, "Unexpected number of statuses"
    assert all(stat == stats[0] for stat in stats), "Statuses are not eq"
    assert len(calls) == 1, "Launcher's `get_status` was called more than once"
    (call,) = calls
    assert call == ((id_,), {}), "IDs were not de-duplicated"
