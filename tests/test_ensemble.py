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

import pytest

from smartsim.entity import _mock
from smartsim.entity._new_ensemble import Ensemble

pytestmark = pytest.mark.group_a

_2x2_PARAMS = {"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
_2_PERM_STRAT = lambda p, n: [{"SPAM": "a", "EGGS": "b"}, {"SPAM": "c", "EGGS": "d"}]
_2x2_EXE_ARG = {"EXE": [["a"],
                        ["b", "c"]],
                "ARGS": [["d"],
                        ["e", "f"]]}


@pytest.fixture
def mock_launcher_settings():
    # TODO: Remove this mock with #587
    #       https://github.com/CrayLabs/SmartSim/pull/587
    return _mock.LaunchSettings()


# fmt: off
@pytest.mark.parametrize(
    "                  params,      strategy,  max_perms, replicas, expected_num_jobs",  # Test Name                                       Misc
    (pytest.param(       None,    "all_perm",          0,        1,                 1 , id="No Parameters or Replicas")                    ,
     pytest.param(_2x2_PARAMS,    "all_perm",          0,        1,                 4 , id="All Permutations")                             ,
     pytest.param(_2x2_PARAMS,        "step",          0,        1,                 2 , id="Stepped Params")                               ,
     pytest.param(_2x2_PARAMS,      "random",          0,        1,                 4 , id="Random Permutations")                          ,
     pytest.param(_2x2_PARAMS,    "all_perm",          1,        1,                 1 , id="All Permutations [Capped Max Permutations]"    , marks=pytest.mark.xfail(reason="'all_perm' strategy currently ignores max number permutations'", strict=True)),
     pytest.param(_2x2_PARAMS,        "step",          1,        1,                 1 , id="Stepped Params [Capped Max Permutations]"      , marks=pytest.mark.xfail(reason="'step' strategy currently ignores max number permutations'", strict=True)),
     pytest.param(_2x2_PARAMS,      "random",          1,        1,                 1 , id="Random Permutations [Capped Max Permutations]"), #     ^^^^^^^^^^^^^^^^^
     pytest.param(         {}, _2_PERM_STRAT,          0,        1,                 2 , id="Custom Permutation Strategy")                  , # TODO: I would argue that we should make these cases pass
     pytest.param(         {},    "all_perm",          0,        5,                 5 , id="Identical Replicas")                           ,
     pytest.param(_2x2_PARAMS,    "all_perm",          0,        2,                 8 , id="Replicas of All Permutations")                 ,
     pytest.param(_2x2_PARAMS,        "step",          0,        2,                 4 , id="Replicas of Stepped Params")                   ,
     pytest.param(_2x2_PARAMS,      "random",          3,        2,                 6 , id="Replicas of Random Permutations")              ,
))
# fmt: on
def test_expected_number_of_apps_created(
    # Parameterized
    params,
    strategy,
    max_perms,
    replicas,
    expected_num_jobs,
    # Other fixtures
    mock_launcher_settings,
):
    jobs = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        file_parameters=params,
        permutation_strategy=strategy,
        max_permutations=max_perms,
        replicas=replicas,
    ).as_jobs(mock_launcher_settings)
    assert len(jobs) == expected_num_jobs


def test_ensemble_without_any_members_rasies_when_cast_to_jobs(mock_launcher_settings):
    with pytest.raises(ValueError):
        Ensemble(
            "test_ensemble",
            "echo",
            ("hello",),
            permutation_strategy=lambda p, n: [{}],
            replicas=0,
        ).as_jobs(mock_launcher_settings)


def test_strategy_error_raised_if_a_strategy_that_dne_is_requested():
    with pytest.raises(ValueError):
        Ensemble(
            "test_ensemble",
            "echo",
            ("hello",),
            permutation_strategy="THIS-STRATEGY-DNE",
        )._create_applications()


@pytest.mark.parametrize(
    "params",
    (
        pytest.param({"SPAM": ["eggs"]}, id="Non-Empty Params"),
        pytest.param({}, id="Empty Params"),
        pytest.param(None, id="Nullish Params"),
    ),
)
def test_replicated_applications_have_eq_deep_copies_of_parameters(params):
    apps = list(
        Ensemble(
            "test_ensemble", "echo", ("hello",), replicas=4, file_parameters=params
        )._create_applications()
    )
    assert len(apps) >= 2  # Sanitiy check to make sure the test is valid
    assert all(app_1.params == app_2.params for app_1 in apps for app_2 in apps)
    assert all(
        app_1.params is not app_2.params
        for app_1 in apps
        for app_2 in apps
        if app_1 is not app_2
    )

def test_plz_work(mock_launcher_settings):
    Ensemble(
            "test_ensemble", "echo", ("hello",), replicas=4, file_parameters=_2x2_PARAMS, exe_arg_parameters=_2x2_EXE_ARG
        )._create_applications()