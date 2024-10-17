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

import contextlib
import itertools

import pytest

from smartsim._core.control.launch_history import LaunchHistory
from smartsim._core.utils.launcher import LauncherProtocol, create_job_id

pytestmark = pytest.mark.group_a


class MockLancher(LauncherProtocol):
    __hash__ = object.__hash__

    @classmethod
    def create(cls, _):
        raise NotImplementedError

    def start(self, _):
        raise NotImplementedError

    def get_status(self, *_):
        raise NotImplementedError

    def stop_jobs(self, *_):
        raise NotImplementedError


LAUNCHER_INSTANCE_A = MockLancher()
LAUNCHER_INSTANCE_B = MockLancher()


@pytest.mark.parametrize(
    "initial_state, to_save",
    (
        pytest.param(
            {},
            [(MockLancher(), create_job_id())],
            id="Empty state, one save",
        ),
        pytest.param(
            {},
            [(MockLancher(), create_job_id()), (MockLancher(), create_job_id())],
            id="Empty state, many save",
        ),
        pytest.param(
            {},
            [
                (LAUNCHER_INSTANCE_A, create_job_id()),
                (LAUNCHER_INSTANCE_A, create_job_id()),
            ],
            id="Empty state, repeat launcher instance",
        ),
        pytest.param(
            {create_job_id(): MockLancher()},
            [(MockLancher(), create_job_id())],
            id="Preexisting state, one save",
        ),
        pytest.param(
            {create_job_id(): MockLancher()},
            [(MockLancher(), create_job_id()), (MockLancher(), create_job_id())],
            id="Preexisting state, many save",
        ),
        pytest.param(
            {create_job_id(): LAUNCHER_INSTANCE_A},
            [(LAUNCHER_INSTANCE_A, create_job_id())],
            id="Preexisting state, repeat launcher instance",
        ),
    ),
)
def test_save_launch(initial_state, to_save):
    history = LaunchHistory(initial_state)
    launcher = MockLancher()

    assert history._id_to_issuer == initial_state
    for launcher, id_ in to_save:
        history.save_launch(launcher, id_)
    assert history._id_to_issuer == initial_state | {id_: l for l, id_ in to_save}


def test_save_launch_raises_if_id_already_in_use():
    launcher = MockLancher()
    other_launcher = MockLancher()
    id_ = create_job_id()
    history = LaunchHistory()
    history.save_launch(launcher, id_)
    with pytest.raises(ValueError):
        history.save_launch(other_launcher, id_)


@pytest.mark.parametrize(
    "ids_to_issuer, expected_num_launchers",
    (
        pytest.param(
            {create_job_id(): MockLancher()},
            1,
            id="One launch, one instance",
        ),
        pytest.param(
            {create_job_id(): LAUNCHER_INSTANCE_A for _ in range(5)},
            1,
            id="Many launch, one instance",
        ),
        pytest.param(
            {create_job_id(): MockLancher() for _ in range(5)},
            5,
            id="Many launch, many instance",
        ),
    ),
)
def test_iter_past_launchers(ids_to_issuer, expected_num_launchers):
    history = LaunchHistory(ids_to_issuer)
    assert len(list(history.iter_past_launchers())) == expected_num_launchers
    known_launchers = set(history._id_to_issuer.values())
    assert all(
        launcher in known_launchers for launcher in history.iter_past_launchers()
    )


ID_A = create_job_id()
ID_B = create_job_id()
ID_C = create_job_id()


@pytest.mark.parametrize(
    "init_state, ids, expected_group_by",
    (
        pytest.param(
            {ID_A: LAUNCHER_INSTANCE_A, ID_B: LAUNCHER_INSTANCE_A},
            None,
            {LAUNCHER_INSTANCE_A: {ID_A, ID_B}},
            id="All known ids, single launcher",
        ),
        pytest.param(
            {ID_A: LAUNCHER_INSTANCE_A, ID_B: LAUNCHER_INSTANCE_A},
            {ID_A},
            {LAUNCHER_INSTANCE_A: {ID_A}},
            id="Subset known ids, single launcher",
        ),
        pytest.param(
            {ID_A: LAUNCHER_INSTANCE_A, ID_B: LAUNCHER_INSTANCE_B},
            None,
            {LAUNCHER_INSTANCE_A: {ID_A}, LAUNCHER_INSTANCE_B: {ID_B}},
            id="All known ids, many launchers",
        ),
        pytest.param(
            {ID_A: LAUNCHER_INSTANCE_A, ID_B: LAUNCHER_INSTANCE_B},
            {ID_A},
            {LAUNCHER_INSTANCE_A: {ID_A}},
            id="Subset known ids, many launchers, same issuer",
        ),
        pytest.param(
            {
                ID_A: LAUNCHER_INSTANCE_A,
                ID_B: LAUNCHER_INSTANCE_B,
                ID_C: LAUNCHER_INSTANCE_A,
            },
            {ID_A, ID_B},
            {LAUNCHER_INSTANCE_A: {ID_A}, LAUNCHER_INSTANCE_B: {ID_B}},
            id="Subset known ids, many launchers, many issuer",
        ),
    ),
)
def test_group_by_launcher(init_state, ids, expected_group_by):
    histroy = LaunchHistory(init_state)
    assert histroy.group_by_launcher(ids) == expected_group_by


@pytest.mark.parametrize(
    "ctx, unknown_ok",
    (
        pytest.param(pytest.raises(ValueError), False, id="unknown_ok=False"),
        pytest.param(contextlib.nullcontext(), True, id="unknown_ok=True"),
    ),
)
def test_group_by_launcher_encounters_unknown_launch_id(ctx, unknown_ok):
    histroy = LaunchHistory()
    with ctx:
        assert histroy.group_by_launcher([create_job_id()], unknown_ok=unknown_ok) == {}
