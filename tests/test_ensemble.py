import pytest

from smartsim.entity import Ensemble, Model
from smartsim.error import SSUnsupportedError, UserStrategyError
from smartsim.settings import RunSettings

"""
Test ensemble creation

TODO: test to add
- test batch settings/run_setting combinations and errors
- test replica creation
"""

# ---- helpers ------------------------------------------------------


def step_values(param_names, param_values):
    permutations = []
    for p in zip(*param_values):
        permutations.append(dict(zip(param_names, p)))
    return permutations


# bad permuation strategy that doesnt return
# a list of dictionaries
def bad_strategy(names, values):
    return -1


# test bad perm strat that returns a list but of lists
# not dictionaries
def bad_strategy_2(names, values):
    return [values]


rs = RunSettings("python", exe_args="sleep.py")

# ----- Test param generation  ----------------------------------------


def test_all_perm():
    """Test all permutation strategy"""
    params = {"h": [5, 6]}
    ensemble = Ensemble("all_perm", params, run_settings=rs, perm_strat="all_perm")
    assert len(ensemble) == 2
    assert ensemble.entities[0].params["h"] == 5
    assert ensemble.entities[1].params["h"] == 6


def test_step():
    """Test step strategy"""
    params = {"h": [5, 6], "g": [7, 8]}
    ensemble = Ensemble("step", params, run_settings=rs, perm_strat="step")
    assert len(ensemble) == 2

    model_1_params = {"h": 5, "g": 7}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"h": 6, "g": 8}
    assert ensemble.entities[1].params == model_2_params


def test_random():
    """Test random strategy"""
    random_ints = [4, 5, 6, 7, 8]
    params = {"h": random_ints}
    ensemble = Ensemble(
        "random_test",
        params,
        run_settings=rs,
        perm_strat="random",
        n_models=len(random_ints),
    )
    assert len(ensemble) == len(random_ints)
    assigned_params = [m.params["h"] for m in ensemble.entities]
    assert all([x in random_ints for x in assigned_params])


def test_user_strategy():
    """Test a user provided strategy"""
    params = {"h": [5, 6], "g": [7, 8]}
    ensemble = Ensemble("step", params, run_settings=rs, perm_strat=step_values)
    assert len(ensemble) == 2

    model_1_params = {"h": 5, "g": 7}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"h": 6, "g": 8}
    assert ensemble.entities[1].params == model_2_params


# ----- Error Handling --------------------------------------

# unknown permuation strategy
def test_unknown_perm_strat():
    bad_strat = "not-a-strategy"
    with pytest.raises(SSUnsupportedError):
        e = Ensemble("ensemble", {}, run_settings=rs, perm_strat=bad_strat)


def test_bad_perm_strat():
    params = {"h": [2, 3]}
    with pytest.raises(UserStrategyError):
        e = Ensemble("ensemble", params, run_settings=rs, perm_strat=bad_strategy)


def test_bad_perm_strat_2():
    params = {"h": [2, 3]}
    with pytest.raises(UserStrategyError):
        e = Ensemble("ensemble", params, run_settings=rs, perm_strat=bad_strategy_2)


# bad argument type in params
def test_incorrect_param_type():
    # can either be a list, str, or int
    params = {"h": {"h": [5]}}
    with pytest.raises(TypeError):
        e = Ensemble("ensemble", params, run_settings=rs)


def test_add_model_type():
    params = {"h": 5}
    e = Ensemble("ensemble", params, run_settings=rs)
    with pytest.raises(TypeError):
        # should be a Model not string
        e.add_model("model")
