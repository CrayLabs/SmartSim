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


from copy import deepcopy

import pytest

from smartsim import Experiment
from smartsim.entity import Ensemble, Model
from smartsim.error import EntityExistsError, SSUnsupportedError, UserStrategyError
from smartsim.settings import RunSettings

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


"""
Test ensemble creation

TODO: test to add
- test batch settings/run_setting combinations and errors
- test replica creation
"""

# ---- helpers ------------------------------------------------------


def step_values(param_names, param_values, n_models=0):
    permutations = []
    for p in zip(*param_values):
        permutations.append(dict(zip(param_names, p)))
    return permutations


# bad permutation strategy that doesn't return
# a list of dictionaries
def bad_strategy(names, values, n_models=0):
    return -1


# test bad perm strategy that returns a list but of lists
# not dictionaries
def bad_strategy_2(names, values, n_models=0):
    return [values]


rs = RunSettings("python", exe_args="sleep.py")

# ----- Test param generation  ----------------------------------------


def test_all_perm():
    """Test all permutation strategy"""
    params = {"h": [5, 6]}
    ensemble = Ensemble("all_perm", params, run_settings=rs, perm_strat="all_perm")
    assert len(ensemble) == 2
    assert ensemble.entities[0].params["h"] == "5"
    assert ensemble.entities[1].params["h"] == "6"


def test_step():
    """Test step strategy"""
    params = {"h": [5, 6], "g": [7, 8]}
    ensemble = Ensemble("step", params, run_settings=rs, perm_strat="step")
    assert len(ensemble) == 2

    model_1_params = {"h": "5", "g": "7"}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"h": "6", "g": "8"}
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
    assert all([int(x) in random_ints for x in assigned_params])

    ensemble = Ensemble(
        "random_test",
        params,
        run_settings=rs,
        perm_strat="random",
        n_models=len(random_ints) - 1,
    )
    assert len(ensemble) == len(random_ints) - 1
    assigned_params = [m.params["h"] for m in ensemble.entities]
    assert all([int(x) in random_ints for x in assigned_params])


def test_user_strategy():
    """Test a user provided strategy"""
    params = {"h": [5, 6], "g": [7, 8]}
    ensemble = Ensemble("step", params, run_settings=rs, perm_strat=step_values)
    assert len(ensemble) == 2

    model_1_params = {"h": "5", "g": "7"}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"h": "6", "g": "8"}
    assert ensemble.entities[1].params == model_2_params


# ----- Model arguments -------------------------------------


def test_arg_params():
    """Test parameterized exe arguments"""
    params = {"H": [5, 6], "g_param": ["a", "b"]}

    # Copy rs to avoid modifying referenced object
    rs_copy = deepcopy(rs)
    rs_orig_args = rs_copy.exe_args
    ensemble = Ensemble(
        "step",
        params=params,
        params_as_args=list(params.keys()),
        run_settings=rs_copy,
        perm_strat="step",
    )
    assert len(ensemble) == 2

    exe_args_0 = rs_orig_args + ["-H", "5", "--g_param=a"]
    assert ensemble.entities[0].run_settings.exe_args == exe_args_0

    exe_args_1 = rs_orig_args + ["-H", "6", "--g_param=b"]
    assert ensemble.entities[1].run_settings.exe_args == exe_args_1


def test_arg_and_model_params_step():
    """Test parameterized exe arguments combined with
    model parameters and step strategy
    """
    params = {"H": [5, 6], "g_param": ["a", "b"], "h": [5, 6], "g": [7, 8]}

    # Copy rs to avoid modifying referenced object
    rs_copy = deepcopy(rs)
    rs_orig_args = rs_copy.exe_args
    ensemble = Ensemble(
        "step",
        params,
        params_as_args=["H", "g_param"],
        run_settings=rs_copy,
        perm_strat="step",
    )
    assert len(ensemble) == 2

    exe_args_0 = rs_orig_args + ["-H", "5", "--g_param=a"]
    assert ensemble.entities[0].run_settings.exe_args == exe_args_0

    exe_args_1 = rs_orig_args + ["-H", "6", "--g_param=b"]
    assert ensemble.entities[1].run_settings.exe_args == exe_args_1

    model_1_params = {"H": "5", "g_param": "a", "h": "5", "g": "7"}
    assert ensemble.entities[0].params == model_1_params

    model_2_params = {"H": "6", "g_param": "b", "h": "6", "g": "8"}
    assert ensemble.entities[1].params == model_2_params


def test_arg_and_model_params_all_perms():
    """Test parameterized exe arguments combined with
    model parameters and all_perm strategy
    """
    params = {"h": [5, 6], "g_param": ["a", "b"]}

    # Copy rs to avoid modifying referenced object
    rs_copy = deepcopy(rs)
    rs_orig_args = rs_copy.exe_args
    ensemble = Ensemble(
        "step",
        params,
        params_as_args=["g_param"],
        run_settings=rs_copy,
        perm_strat="all_perm",
    )
    assert len(ensemble) == 4

    exe_args_0 = rs_orig_args + ["--g_param=a"]
    assert ensemble.entities[0].run_settings.exe_args == exe_args_0
    assert ensemble.entities[2].run_settings.exe_args == exe_args_0

    exe_args_1 = rs_orig_args + ["--g_param=b"]
    assert ensemble.entities[1].run_settings.exe_args == exe_args_1
    assert ensemble.entities[3].run_settings.exe_args == exe_args_1

    model_0_params = {"g_param": "a", "h": "5"}
    assert ensemble.entities[0].params == model_0_params
    model_1_params = {"g_param": "b", "h": "5"}
    assert ensemble.entities[1].params == model_1_params
    model_2_params = {"g_param": "a", "h": "6"}
    assert ensemble.entities[2].params == model_2_params
    model_3_params = {"g_param": "b", "h": "6"}
    assert ensemble.entities[3].params == model_3_params


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


def test_add_existing_model():
    params_1 = {"h": 5}
    params_2 = {"z": 6}
    model_1 = Model("identical_name", params_1, "", rs)
    model_2 = Model("identical_name", params_2, "", rs)
    e = Ensemble("ensemble", params_1, run_settings=rs)
    e.add_model(model_1)
    with pytest.raises(EntityExistsError):
        e.add_model(model_2)


# ----- Other --------------------------------------


def test_models_property():
    params = {"h": [5, 6, 7, 8]}
    e = Ensemble("test", params, run_settings=rs)
    models = e.models
    assert models == [model for model in e]


def test_key_prefixing():
    params_1 = {"h": [5, 6, 7, 8]}
    params_2 = {"z": 6}
    e = Ensemble("test", params_1, run_settings=rs)
    model = Model("model", params_2, "", rs)
    e.add_model(model)
    assert e.query_key_prefixing() == False
    e.enable_key_prefixing()
    assert e.query_key_prefixing() == True


def test_ensemble_type():
    exp = Experiment("name")
    ens_settings = RunSettings("python")
    ensemble = exp.create_ensemble("name", replicas=4, run_settings=ens_settings)
    assert ensemble.type == "Ensemble"
