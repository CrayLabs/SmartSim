import pytest
from smartsim.entity import Ensemble, Model
from smartsim.error import UserStrategyError, SSConfigError
from smartsim.error import EntityExistsError, SSUnsupportedError

# ----- Test ------------------------------------------------

rs = {"executable": "python"}


def test_empty_ensemble():
    ensemble = Ensemble("empty", {}, "/", run_settings=rs)
    assert len(ensemble) == 0


def test_all_perm():
    params = {"h": [5, 6]}
    ensemble = Ensemble(
        "all_perm",
        params,
        "/",  # path not needed
        perm_strat="all_perm",
        run_settings=rs,
    )
    assert len(ensemble) == 2
    assert ensemble.entities[0].params["h"] == 5
    assert ensemble.entities[1].params["h"] == 6


def test_step():
    params = {"h": [5, 6], "g": [7, 8]}
    ensemble = Ensemble(
        "step", params, "/", perm_strat="step", run_settings=rs  # path not needed
    )
    assert len(ensemble) == 2

    model_1_params = {"h": 5, "g": 7}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"h": 6, "g": 8}
    assert ensemble.entities[1].params == model_2_params


def test_random():
    random_ints = [4, 5, 6, 7, 8]
    params = {"h": random_ints}
    ensemble = Ensemble(
        "random_test",
        params,
        "/",  # path not needed
        perm_strat="random",
        run_settings=rs,
        n_models=len(random_ints),
    )
    assert len(ensemble) == len(random_ints)
    assigned_params = [m.params["h"] for m in ensemble.entities]
    assert all([x in random_ints for x in assigned_params])


def step_values(param_names, param_values):
    permutations = []
    for p in zip(*param_values):
        permutations.append(dict(zip(param_names, p)))
    return permutations


def test_user_strategy():
    params = {"h": [5, 6], "g": [7, 8]}
    ensemble = Ensemble(
        "step", params, "/", perm_strat=step_values, run_settings=rs  # path not needed
    )
    assert len(ensemble) == 2

    model_1_params = {"h": 5, "g": 7}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"h": 6, "g": 8}
    assert ensemble.entities[1].params == model_2_params


# ----- Error Handling --------------------------------------

# add model that already exists
def test_add_existing_model():
    m = Model("model", {}, "/", rs)
    ensemble = Ensemble("ensemble", {}, "/", run_settings=rs)
    ensemble.add_model(m)
    with pytest.raises(EntityExistsError):
        ensemble.add_model(m)


# unknown permuation strategy
def test_unknown_perm_strat():
    bad_strat = "not-a-strategy"
    with pytest.raises(SSUnsupportedError):
        e = Ensemble("ensemble", {}, "/", run_settings=rs, perm_strat=bad_strat)


# bad permuation strategy that doesnt return
# a list of dictionaries
def bad_strategy(names, values):
    return -1


def test_bad_perm_strat():
    params = {"h": [2, 3]}
    with pytest.raises(UserStrategyError):
        e = Ensemble("ensemble", params, "/", run_settings=rs, perm_strat=bad_strategy)


# test bad perm strat that returns a list but of lists
# not dictionaries
def bad_strategy_2(names, values):
    return [values]


def test_bad_perm_strat_2():
    params = {"h": [2, 3]}
    with pytest.raises(UserStrategyError):
        e = Ensemble(
            "ensemble", params, "/", run_settings=rs, perm_strat=bad_strategy_2
        )


# bad argument type in params
def test_incorrect_param_type():
    # can either be a list, str, or int
    params = {"h": {"h": [5]}}
    with pytest.raises(TypeError):
        e = Ensemble("ensemble", params, "/", run_settings=rs)


# no exe
def test_no_executable():
    params = {"h": [2, 3]}
    with pytest.raises(SSConfigError):
        e = Ensemble("ensemble", params, "/", {})


# invalid/non-existant exe
def test_bad_executable():
    run_set = {"executable": "not-an-exe"}
    params = {"h": [2, 3]}
    with pytest.raises(SSConfigError):
        e = Ensemble("ensemble", params, "/", run_settings=run_set)