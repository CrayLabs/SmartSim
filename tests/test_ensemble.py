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

import os
import pathlib
import typing as t
from glob import glob
from os import path as osp

import pytest

from smartsim._core.generation.operations.ensemble_operations import (
    EnsembleConfigureOperation,
    EnsembleCopyOperation,
    EnsembleSymlinkOperation,
)
from smartsim.builders.ensemble import Ensemble, FileSet
from smartsim.builders.utils import strategies
from smartsim.builders.utils.strategies import ParamSet
from smartsim.entity import Application
from smartsim.settings.launch_settings import LaunchSettings

pytestmark = pytest.mark.group_a

_2x2_PARAMS = {"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
_2x2_EXE_ARG = {"EXE": [["a"], ["b", "c"]], "ARGS": [["d"], ["e", "f"]]}


def user_created_function(
    file_params: t.Mapping[str, t.Sequence[str]],
    exe_arg_params: t.Mapping[str, t.Sequence[t.Sequence[str]]],
    n_permutations: int = 0,
) -> list[ParamSet]:
    return [ParamSet({}, {})]


@pytest.fixture
def ensemble():
    return Ensemble(
        name="ensemble_name",
        exe="python",
        exe_args="sleepy.py",
        exe_arg_parameters={"-N": 2},
        permutation_strategy="all_perm",
        max_permutations=2,
    )


@pytest.fixture
def mock_launcher_settings(wlmutils):
    return LaunchSettings(wlmutils.get_test_launcher(), {}, {})


def test_ensemble_init():
    """Validate Ensemble init"""
    ensemble = Ensemble(name="ensemble_name", exe="python")
    assert isinstance(ensemble, Ensemble)
    assert ensemble.name == "ensemble_name"
    assert ensemble.exe == os.fspath("python")


def test_ensemble_init_empty_params() -> None:
    """Invalid Ensemble init"""
    with pytest.raises(TypeError):
        Ensemble()


def test_exe_property(ensemble):
    """Validate Ensemble exe property"""
    exe = ensemble.exe
    assert exe == ensemble.exe


@pytest.mark.parametrize(
    "exe,error",
    (
        pytest.param(123, TypeError, id="exe as integer"),
        pytest.param(None, TypeError, id="exe as None"),
    ),
)
def test_set_exe_invalid(ensemble, exe, error):
    """Validate Ensemble exe setter throws"""
    with pytest.raises(error):
        ensemble.exe = exe


@pytest.mark.parametrize(
    "exe",
    (
        pytest.param(pathlib.Path("this/is/path"), id="exe as pathlib"),
        pytest.param("this/is/path", id="exe as str"),
    ),
)
def test_set_exe_valid(ensemble, exe):
    """Validate Ensemble exe setter sets"""
    ensemble.exe = exe
    assert ensemble.exe == str(exe)


def test_exe_args_property(ensemble):
    """Validate Ensemble exe_args property"""
    exe_args = ensemble.exe_args
    assert exe_args == ensemble.exe_args


@pytest.mark.parametrize(
    "exe_args,error",
    (
        pytest.param(123, TypeError, id="exe_args as integer"),
        pytest.param(None, TypeError, id="exe_args as None"),
        pytest.param(["script.py", 123], TypeError, id="exe_args as None"),
    ),
)
def test_set_exe_args_invalid(ensemble, exe_args, error):
    """Validate Ensemble exe_arg setter throws"""
    with pytest.raises(error):
        ensemble.exe_args = exe_args


@pytest.mark.parametrize(
    "exe_args",
    (
        pytest.param(["script.py", "another.py"], id="exe_args as pathlib"),
        pytest.param([], id="exe_args as str"),
    ),
)
def test_set_exe_args_valid(ensemble, exe_args):
    """Validate Ensemble exe_args setter sets"""
    ensemble.exe_args = exe_args
    assert ensemble.exe_args == exe_args


def test_exe_arg_parameters_property(ensemble):
    """Validate Ensemble exe_arg_parameters property"""
    exe_arg_parameters = ensemble.exe_arg_parameters
    assert exe_arg_parameters == ensemble.exe_arg_parameters


@pytest.mark.parametrize(
    "exe_arg_parameters",
    (
        pytest.param(
            {"key": [["test"]]},
            id="Value is a sequence of sequence of str",
        ),
        pytest.param(
            {"key": [["test"], ["test"]]},
            id="Value is a sequence of sequence of str",
        ),
    ),
)
def test_set_exe_arg_parameters_valid(exe_arg_parameters, ensemble):
    """Validate Ensemble exe_arg_parameters setter sets"""
    ensemble.exe_arg_parameters = exe_arg_parameters
    assert ensemble.exe_arg_parameters == exe_arg_parameters


@pytest.mark.parametrize(
    "exe_arg_params",
    (
        pytest.param(["invalid"], id="Not a mapping"),
        pytest.param({"key": [1, 2, 3]}, id="Value is not sequence of sequences"),
        pytest.param(
            {"key": [[1, 2, 3], [4, 5, 6]]},
            id="Value is not sequence of sequence of str",
        ),
        pytest.param(
            {1: 2},
            id="key and value wrong type",
        ),
        pytest.param({"1": 2}, id="Value is not mapping of str and str"),
        pytest.param({1: "2"}, id="Key is not str"),
        pytest.param({1: 2}, id="Values not mapping of str and str"),
    ),
)
def test_set_exe_arg_parameters_invalid(exe_arg_params, ensemble):
    """Validate Ensemble exe_arg_parameters setter throws"""
    with pytest.raises(
        TypeError,
        match="exe_arg_parameters argument was not of type mapping "
        "of str and sequences of sequences of strings",
    ):
        ensemble.exe_arg_parameters = exe_arg_params


def test_permutation_strategy_property(ensemble):
    """Validate Ensemble permutation_strategy property"""
    permutation_strategy = ensemble.permutation_strategy
    assert permutation_strategy == ensemble.permutation_strategy


def test_permutation_strategy_set_invalid(ensemble):
    """Validate Ensemble permutation_strategy setter throws"""
    with pytest.raises(
        TypeError,
        match="permutation_strategy argument was not of "
        "type str or PermutationStrategyType",
    ):
        ensemble.permutation_strategy = 2


@pytest.mark.parametrize(
    "strategy",
    (
        pytest.param("all_perm", id="strategy as all_perm"),
        pytest.param("step", id="strategy as step"),
        pytest.param("random", id="strategy as random"),
        pytest.param(user_created_function, id="strategy as user_created_function"),
    ),
)
def test_permutation_strategy_set_valid(ensemble, strategy):
    """Validate Ensemble permutation_strategy setter sets"""
    ensemble.permutation_strategy = strategy
    assert ensemble.permutation_strategy == strategy


def test_max_permutations_property(ensemble):
    """Validate Ensemble max_permutations property"""
    max_permutations = ensemble.max_permutations
    assert max_permutations == ensemble.max_permutations


@pytest.mark.parametrize(
    "max_permutations",
    (
        pytest.param(123, id="max_permutations as str"),
        pytest.param(-1, id="max_permutations as float"),
    ),
)
def test_max_permutations_set_valid(ensemble, max_permutations):
    """Validate Ensemble max_permutations setter sets"""
    ensemble.max_permutations = max_permutations
    assert ensemble.max_permutations == max_permutations


@pytest.mark.parametrize(
    "max_permutations,error",
    (
        pytest.param("str", TypeError, id="max_permutations as str"),
        pytest.param(None, TypeError, id="max_permutations as None"),
        pytest.param(0.1, TypeError, id="max_permutations as float"),
    ),
)
def test_max_permutations_set_invalid(ensemble, max_permutations, error):
    """Validate Ensemble exe_arg setter throws"""
    with pytest.raises(error):
        ensemble.max_permutations = max_permutations


def test_replicas_property(ensemble):
    """Validate Ensemble replicas property"""
    replicas = ensemble.replicas
    assert replicas == ensemble.replicas


@pytest.mark.parametrize(
    "replicas",
    (pytest.param(123, id="replicas as str"),),
)
def test_replicas_set_valid(ensemble, replicas):
    """Validate Ensemble replicas setter sets"""
    ensemble.replicas = replicas
    assert ensemble.replicas == replicas


@pytest.mark.parametrize(
    "replicas,error",
    (
        pytest.param("str", TypeError, id="replicas as str"),
        pytest.param(None, TypeError, id="replicas as None"),
        pytest.param(0.1, TypeError, id="replicas as float"),
        pytest.param(-1, ValueError, id="replicas as negative int"),
    ),
)
def test_replicas_set_invalid(ensemble, replicas, error):
    """Validate Ensemble replicas setter throws"""
    with pytest.raises(error):
        ensemble.replicas = replicas


@pytest.mark.parametrize(
    "                        params,      exe_arg_params,   max_perms,     strategy, expected_combinations",
    (
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            8,
            "all_perm",
            8,
            id="Limit number of perms - 8 : all_perm",
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            1,
            "all_perm",
            1,
            id="Limit number of perms - 1 : all_perm",
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            -1,
            "all_perm",
            16,
            id="All permutations : all_perm",
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            30,
            "all_perm",
            16,
            id="Greater number of perms : all_perm",
        ),
        pytest.param(
            _2x2_PARAMS, {}, -1, "all_perm", 4, id="Empty exe args params : all_perm"
        ),
        pytest.param(
            {}, _2x2_EXE_ARG, -1, "all_perm", 4, id="Empty file params : all_perm"
        ),
        pytest.param(
            {},
            {},
            -1,
            "all_perm",
            1,
            id="Empty exe args params and file params : all_perm",
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            2,
            "step",
            2,
            id="Limit number of perms - 2 : step",
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            1,
            "step",
            1,
            id="Limit number of perms - 1 : step",
        ),
        pytest.param(
            _2x2_PARAMS, _2x2_EXE_ARG, -1, "step", 2, id="All permutations : step"
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            30,
            "step",
            2,
            id="Greater number of perms : step",
        ),
        pytest.param(_2x2_PARAMS, {}, -1, "step", 1, id="Empty exe args params : step"),
        pytest.param({}, _2x2_EXE_ARG, -1, "step", 1, id="Empty file params : step"),
        pytest.param(
            {}, {}, -1, "step", 1, id="Empty exe args params and file params : step"
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            8,
            "random",
            8,
            id="Limit number of perms - 8 : random",
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            1,
            "random",
            1,
            id="Limit number of perms - 1 : random",
        ),
        pytest.param(
            _2x2_PARAMS, _2x2_EXE_ARG, -1, "random", 16, id="All permutations : random"
        ),
        pytest.param(
            _2x2_PARAMS,
            _2x2_EXE_ARG,
            30,
            "random",
            16,
            id="Greater number of perms : random",
        ),
        pytest.param(
            _2x2_PARAMS, {}, -1, "random", 4, id="Empty exe args params : random"
        ),
        pytest.param(
            {}, _2x2_EXE_ARG, -1, "random", 4, id="Empty file params : random"
        ),
        pytest.param(
            {}, {}, -1, "random", 1, id="Empty exe args params and file params : random"
        ),
    ),
)
def test_permutate_file_parameters(
    params, exe_arg_params, max_perms, strategy, expected_combinations
):
    """Test Ensemble._permutate_file_parameters returns expected number of combinations for step, random and all_perm"""
    ensemble = Ensemble(
        "name",
        "echo",
        exe_arg_parameters=exe_arg_params,
        permutation_strategy=strategy,
        max_permutations=max_perms,
    )
    permutation_strategy = strategies.resolve(strategy)
    config_file = EnsembleConfigureOperation(
        src=pathlib.Path("/src"), file_parameters=params
    )
    file_set_list = ensemble._permutate_file_parameters(
        config_file, permutation_strategy
    )
    assert len(file_set_list) == expected_combinations
    for file_set in file_set_list:
        assert isinstance(file_set, FileSet)
        assert file_set.file == config_file


def test_cartesian_values():
    """Test Ensemble._cartesian_values returns expected number of combinations"""
    ensemble = Ensemble(
        "name",
        "echo",
        exe_arg_parameters={"-N": ["1", "2"]},
    )
    permutation_strategy = strategies.resolve("all_perm")
    config_file_1 = EnsembleConfigureOperation(
        src=pathlib.Path("/src_1"), file_parameters={"SPAM": ["a"]}
    )
    config_file_2 = EnsembleConfigureOperation(
        src=pathlib.Path("/src_2"), file_parameters={"EGGS": ["b"]}
    )
    file_set_list = []
    file_set_list.append(
        ensemble._permutate_file_parameters(config_file_1, permutation_strategy)
    )
    file_set_list.append(
        ensemble._permutate_file_parameters(config_file_2, permutation_strategy)
    )
    file_set_tuple = ensemble._cartesian_values(file_set_list)
    assert isinstance(file_set_tuple, list)
    assert len(file_set_tuple) == 4
    for tup in file_set_tuple:
        assert isinstance(tup, tuple)
        assert len(tup) == 2
        files = [fs.file for fs in tup]
        # Validate that each config file is in the tuple of FileSets
        assert config_file_1 in files
        assert config_file_2 in files


def test_attach_files():
    ensemble = Ensemble("mock_ensemble", "echo")
    ensemble.files.add_copy(src=pathlib.Path("/copy"))
    ensemble.files.add_symlink(src=pathlib.Path("/symlink"))
    app = Application("mock_app", "echo")
    file_set_1 = FileSet(
        EnsembleConfigureOperation(
            src=pathlib.Path("/src_1"), file_parameters={"FOO": "TOE"}
        ),
        ParamSet({}, {}),
    )
    file_set_2 = FileSet(
        EnsembleConfigureOperation(
            src=pathlib.Path("/src_2"), file_parameters={"FOO": "TOE"}
        ),
        ParamSet({}, {}),
    )
    ensemble._attach_files(app, (file_set_1, file_set_2))
    assert len(app.files.copy_operations) == 1
    assert len(app.files.symlink_operations) == 1
    assert len(app.files.configure_operations) == 2
    sources = [file.src for file in app.files.configure_operations]
    assert file_set_1.file.src in sources
    assert file_set_2.file.src in sources


# fmt: off
@pytest.mark.parametrize(
    "                  params,      exe_arg_params,   max_perms, replicas, expected_num_jobs",
    (pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          30,        1,                16 , id="Set max permutation high"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          -1,        1,                16 , id="Set max permutation negative"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           0,        1,                 1 , id="Set max permutation zero"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           1,        1,                 1 , id="Set max permutation to lowest"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           6,        2,                12 , id="Set max permutation, set replicas"),
     pytest.param(         {},        _2x2_EXE_ARG,           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(_2x2_PARAMS,                  {},           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(         {},                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas")
))
# fmt: on
def test_all_perm_strategy_create_application(
    # Parameterized
    params,
    exe_arg_params,
    max_perms,
    replicas,
    expected_num_jobs,
):
    e = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        exe_arg_parameters=exe_arg_params,
        permutation_strategy="all_perm",
        max_permutations=max_perms,
        replicas=replicas,
    )
    e.files.add_configuration(src=pathlib.Path("/src_1"), file_parameters=params)
    apps = e._create_applications()
    assert len(apps) == expected_num_jobs


# fmt: off
@pytest.mark.parametrize(
    "                  params,      exe_arg_params,   max_perms, replicas, expected_num_jobs",         
    (pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          30,        1,                 2 , id="Set max permutation high"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          -1,        1,                 2 , id="Set max permutation negtive"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           0,        1,                 1 , id="Set max permutation zero"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           1,        1,                 1 , id="Set max permutation to lowest"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           6,        2,                 4 , id="Set max permutation, set replicas"),
     pytest.param(         {},        _2x2_EXE_ARG,           6,        2,                 2 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(_2x2_PARAMS,                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(         {},                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas")
))
# fmt: on
def test_step_strategy_create_application(
    # Parameterized
    params,
    exe_arg_params,
    max_perms,
    replicas,
    expected_num_jobs,
):
    e = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        exe_arg_parameters=exe_arg_params,
        permutation_strategy="step",
        max_permutations=max_perms,
        replicas=replicas,
    )
    e.files.add_configuration(src=pathlib.Path("/src_1"), file_parameters=params)
    apps = e._create_applications()
    assert len(apps) == expected_num_jobs


# fmt: off
@pytest.mark.parametrize(
    "                  params,      exe_arg_params,   max_perms, replicas, expected_num_jobs",         
    (pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          30,        1,                16 , id="Set max permutation high"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          -1,        1,                16 , id="Set max permutation negative"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           0,        1,                 1 , id="Set max permutation zero"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           1,        1,                 1 , id="Set max permutation to lowest"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           6,        2,                12 , id="Set max permutation, set replicas"),
     pytest.param(         {},        _2x2_EXE_ARG,           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(_2x2_PARAMS,                  {},           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(         {},                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas")
))
# fmt: on
def test_random_strategy_create_application(
    # Parameterized
    params,
    exe_arg_params,
    max_perms,
    replicas,
    expected_num_jobs,
):
    e = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        exe_arg_parameters=exe_arg_params,
        permutation_strategy="random",
        max_permutations=max_perms,
        replicas=replicas,
    )
    e.files.add_configuration(src=pathlib.Path("/src_1"), file_parameters=params)
    apps = e._create_applications()
    assert len(apps) == expected_num_jobs


def test_ensemble_type_build_jobs():
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    with pytest.raises(TypeError):
        ensemble.build_jobs("invalid")


@pytest.mark.parametrize(
    "bad_settings",
    [pytest.param(None, id="Nullish"), pytest.param("invalid", id="String")],
)
def test_ensemble_incorrect_launch_settings_type(bad_settings):
    """test starting an ensemble with invalid launch settings"""
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    with pytest.raises(TypeError):
        ensemble.build_jobs(bad_settings)


def test_ensemble_user_created_strategy(mock_launcher_settings, test_dir):
    jobs = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        permutation_strategy=user_created_function,
    ).build_jobs(mock_launcher_settings)
    assert len(jobs) == 1


def test_ensemble_without_any_members_raises_when_cast_to_jobs(mock_launcher_settings):
    with pytest.raises(ValueError):
        Ensemble(
            "test_ensemble",
            "echo",
            ("hello", "world"),
            permutation_strategy="random",
            max_permutations=30,
            replicas=0,
        ).build_jobs(mock_launcher_settings)


def test_strategy_error_raised_if_a_strategy_that_dne_is_requested(test_dir):
    with pytest.raises(ValueError):
        Ensemble(
            "test_ensemble",
            "echo",
            ("hello",),
            permutation_strategy="THIS-STRATEGY-DNE",
        )._create_applications()
