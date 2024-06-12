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

from smartsim.entity import strategies
from smartsim.entity.param_data_class import ParamSet
from smartsim.error import errors

pytestmark = pytest.mark.group_a


def test_strategy_registration(monkeypatch):
    monkeypatch.setattr(strategies, "_REGISTERED_STRATEGIES", {})
    assert not strategies._REGISTERED_STRATEGIES

    new_strat = lambda params, nmax: []
    strategies._register("new_strat")(new_strat)
    assert strategies._REGISTERED_STRATEGIES == {"new_strat": new_strat}


def test_strategies_cannot_be_overwritten(monkeypatch):
    monkeypatch.setattr(
        strategies, "_REGISTERED_STRATEGIES", {"some-strategy": lambda params, nmax: []}
    )
    with pytest.raises(ValueError):
        strategies._register("some-strategy")(lambda params, nmax: [])


def test_strategy_retreval(monkeypatch):
    new_strat_a = lambda params, nmax: []
    new_strat_b = lambda params, nmax: []

    monkeypatch.setattr(
        strategies,
        "_REGISTERED_STRATEGIES",
        {"new_strat_a": new_strat_a, "new_strat_b": new_strat_b},
    )
    assert strategies.resolve("new_strat_a") == new_strat_a
    assert strategies.resolve("new_strat_b") == new_strat_b


def test_user_strategy_error_raised_when_attempting_to_get_unknown_strat():
    with pytest.raises(ValueError):
        strategies.resolve("NOT-REGISTERED")


def broken_strategy(p, n, e):
    raise Exception("This custom strategy raised an error")


@pytest.mark.parametrize(
    "strategy",
    (
        pytest.param(broken_strategy, id="Strategy raises during execution"),
        pytest.param(lambda params, nmax: 123, id="Does not return a list"),
        pytest.param(
            lambda params, nmax: [1, 2, 3], id="Does not return a list of dicts"
        ),
    ),
)
def test_custom_strategy_raises_user_strategy_error_if_something_goes_wrong(strategy):
    with pytest.raises(errors.UserStrategyError):
        strategies.resolve(strategy)({"SPAM": ["EGGS"]}, 123)


@pytest.mark.parametrize(
    "strategy, expected_output",
    (
        pytest.param(
            strategies.create_all_permutations,
            (
                [
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "c"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "c"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "d"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "d"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "c"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "c"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "d"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "d"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                ]
            ),
            id="All Permutations",
        ),
        pytest.param(
            strategies.step_values,
            (
                [
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "c"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "d"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                ]
            ),
            id="Step Values",
        ),
        pytest.param(
            strategies.random_permutations,
            (
                [
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "c"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "c"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "d"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "a", "EGGS": "d"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "c"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "c"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "d"}, _exe_args={"EXE": ["a"]}
                    ),
                    ParamSet(
                        _params={"SPAM": "b", "EGGS": "d"},
                        _exe_args={"EXE": ["b", "c"]},
                    ),
                ]
            ),
            id="Uncapped Random Permutations",
        ),
    ),
)
def test_strategy_returns_expected_set(strategy, expected_output):
    params = {"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
    exe_args = {"EXE": [["a"], ["b", "c"]]}
    output = list(strategy(params, exe_args, 50))
    assert len(output) == len(expected_output)
    assert all(item in expected_output for item in output)
    assert all(item in output for item in expected_output)
