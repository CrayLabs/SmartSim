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

import typing as t
from glob import glob
from os import path as osp

import pytest

from smartsim.builders.ensemble import Ensemble
from smartsim.builders.utils.strategies import ParamSet
from smartsim.entity.files import EntityFiles
from smartsim.settings.launch_settings import LaunchSettings

pytestmark = pytest.mark.group_a

_2x2_PARAMS = {"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
_2x2_EXE_ARG = {"EXE": [["a"], ["b", "c"]], "ARGS": [["d"], ["e", "f"]]}


@pytest.fixture
def get_gen_configure_dir(fileutils):
    yield fileutils.get_test_conf_path(osp.join("generator_files", "tag_dir_template"))


def user_created_function(
    file_params: t.Mapping[str, t.Sequence[str]],
    exe_arg_params: t.Mapping[str, t.Sequence[t.Sequence[str]]],
    n_permutations: int = 0,
) -> list[ParamSet]:
    return [ParamSet({}, {})]


@pytest.fixture
def mock_launcher_settings(wlmutils):
    return LaunchSettings(wlmutils.get_test_launcher(), {}, {})


def test_exe_property():
    e = Ensemble(name="test", exe="path/to/example_simulation_program")
    exe = e.exe
    assert exe == e.exe


def test_exe_args_property():
    e = Ensemble("test", exe="path/to/example_simulation_program", exe_args="sleepy.py")
    exe_args = e.exe_args
    assert exe_args == e.exe_args


def test_exe_arg_parameters_property():
    exe_arg_parameters = {"-N": 2}
    e = Ensemble(
        "test",
        exe="path/to/example_simulation_program",
        exe_arg_parameters=exe_arg_parameters,
    )
    exe_arg_parameters = e.exe_arg_parameters
    assert exe_arg_parameters == e.exe_arg_parameters


def test_files_property(get_gen_configure_dir):
    tagged_files = sorted(glob(get_gen_configure_dir + "/*"))
    files = EntityFiles(tagged=tagged_files)
    e = Ensemble("test", exe="path/to/example_simulation_program", files=files)
    files = e.files
    assert files == e.files


def test_file_parameters_property():
    file_parameters = {"h": [5, 6, 7, 8]}
    e = Ensemble(
        "test",
        exe="path/to/example_simulation_program",
        file_parameters=file_parameters,
    )
    file_parameters = e.file_parameters
    assert file_parameters == e.file_parameters


def test_ensemble_init_empty_params(test_dir: str) -> None:
    """Ensemble created without required args"""
    with pytest.raises(TypeError):
        Ensemble()


@pytest.mark.parametrize(
    "bad_settings",
    [pytest.param(None, id="Nullish"), pytest.param("invalid", id="String")],
)
def test_ensemble_incorrect_launch_settings_type(bad_settings):
    """test starting an ensemble with invalid launch settings"""
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    with pytest.raises(TypeError):
        ensemble.build_jobs(bad_settings)


def test_ensemble_type_exe():
    ensemble = Ensemble(
        "ensemble-name",
        exe="valid",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError, match="exe argument was not of type str or PathLike str"
    ):
        ensemble.exe = 2


@pytest.mark.parametrize(
    "bad_settings",
    [
        pytest.param([1, 2, 3], id="sequence of ints"),
        pytest.param(0, id="null"),
        pytest.param({"foo": "bar"}, id="dict"),
    ],
)
def test_ensemble_type_exe_args(bad_settings):
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
    )
    with pytest.raises(
        TypeError, match="exe_args argument was not of type sequence of str"
    ):
        ensemble.exe_args = bad_settings


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
def test_ensemble_type_exe_arg_parameters(exe_arg_params):
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="exe_arg_parameters argument was not of type mapping "
        "of str and sequences of sequences of strings",
    ):
        ensemble.exe_arg_parameters = exe_arg_params


def test_ensemble_type_files():
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(TypeError, match="files argument was not of type EntityFiles"):
        ensemble.files = 2


@pytest.mark.parametrize(
    "file_params",
    (
        pytest.param(["invalid"], id="Not a mapping"),
        pytest.param({"key": [1, 2, 3]}, id="Key is not sequence of sequences"),
    ),
)
def test_ensemble_type_file_parameters(file_params):
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="file_parameters argument was not of type "
        "mapping of str and sequence of str",
    ):
        ensemble.file_parameters = file_params


def test_ensemble_type_permutation_strategy():
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="permutation_strategy argument was not of "
        "type str or PermutationStrategyType",
    ):
        ensemble.permutation_strategy = 2


def test_ensemble_type_max_permutations():
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="max_permutations argument was not of type int",
    ):
        ensemble.max_permutations = "invalid"


def test_ensemble_type_replicas():
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        TypeError,
        match="replicas argument was not of type int",
    ):
        ensemble.replicas = "invalid"


def test_ensemble_type_replicas_negative():
    ensemble = Ensemble(
        "ensemble-name",
        exe="echo",
        exe_args=["spam", "eggs"],
    )
    with pytest.raises(
        ValueError,
        match="Number of replicas must be a positive integer",
    ):
        ensemble.replicas = -20


def test_ensemble_type_build_jobs():
    ensemble = Ensemble("ensemble-name", "echo", replicas=2)
    with pytest.raises(TypeError):
        ensemble.build_jobs("invalid")


def test_ensemble_user_created_strategy(mock_launcher_settings, test_dir):
    jobs = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        permutation_strategy=user_created_function,
    ).build_jobs(mock_launcher_settings)
    assert len(jobs) == 1


def test_ensemble_without_any_members_raises_when_cast_to_jobs(
    mock_launcher_settings, test_dir
):
    with pytest.raises(ValueError):
        Ensemble(
            "test_ensemble",
            "echo",
            ("hello", "world"),
            file_parameters=_2x2_PARAMS,
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


@pytest.mark.parametrize(
    "file_parameters",
    (
        pytest.param({"SPAM": ["eggs"]}, id="Non-Empty Params"),
        pytest.param({}, id="Empty Params"),
        pytest.param(None, id="Nullish Params"),
    ),
)
def test_replicated_applications_have_eq_deep_copies_of_parameters(
    file_parameters, test_dir
):
    apps = list(
        Ensemble(
            "test_ensemble",
            "echo",
            ("hello",),
            replicas=4,
            file_parameters=file_parameters,
        )._create_applications()
    )
    assert len(apps) >= 2  # Sanitiy check to make sure the test is valid
    assert all(
        app_1.file_parameters == app_2.file_parameters
        for app_1 in apps
        for app_2 in apps
    )
    assert all(
        app_1.file_parameters is not app_2.file_parameters
        for app_1 in apps
        for app_2 in apps
        if app_1 is not app_2
    )


# fmt: off
@pytest.mark.parametrize(
    "                  params,      exe_arg_params,   max_perms, replicas, expected_num_jobs",         
    (pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          30,        1,                16 , id="Set max permutation high"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          -1,        1,                16 , id="Set max permutation negative"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           0,        1,                 1 , id="Set max permutation zero"),
     pytest.param(_2x2_PARAMS,                None,           4,        1,                 4 , id="No exe arg params or Replicas"),
     pytest.param(       None,        _2x2_EXE_ARG,           4,        1,                 4 , id="No Parameters or Replicas"),
     pytest.param(       None,                None,           4,        1,                 1 , id="No Parameters, Exe_Arg_Param or Replicas"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           1,        1,                 1 , id="Set max permutation to lowest"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           6,        2,                12 , id="Set max permutation, set replicas"),
     pytest.param(         {},        _2x2_EXE_ARG,           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(_2x2_PARAMS,                  {},           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(         {},                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas")
))
# fmt: on
def test_all_perm_strategy(
    # Parameterized
    params,
    exe_arg_params,
    max_perms,
    replicas,
    expected_num_jobs,
    # Other fixtures
    mock_launcher_settings,
    test_dir,
):
    jobs = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        file_parameters=params,
        exe_arg_parameters=exe_arg_params,
        permutation_strategy="all_perm",
        max_permutations=max_perms,
        replicas=replicas,
    ).build_jobs(mock_launcher_settings)
    assert len(jobs) == expected_num_jobs


def test_all_perm_strategy_contents(mock_launcher_settings):
    jobs = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        file_parameters=_2x2_PARAMS,
        exe_arg_parameters=_2x2_EXE_ARG,
        permutation_strategy="all_perm",
        max_permutations=16,
        replicas=1,
    ).build_jobs(mock_launcher_settings)
    assert len(jobs) == 16


# fmt: off
@pytest.mark.parametrize(
    "                  params,      exe_arg_params,   max_perms, replicas, expected_num_jobs",         
    (pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          30,        1,                 2 , id="Set max permutation high"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          -1,        1,                 2 , id="Set max permutation negtive"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           0,        1,                 1 , id="Set max permutation zero"),
     pytest.param(_2x2_PARAMS,                None,           4,        1,                 1 , id="No exe arg params or Replicas"),
     pytest.param(       None,        _2x2_EXE_ARG,           4,        1,                 1 , id="No Parameters or Replicas"),
     pytest.param(       None,                None,           4,        1,                 1 , id="No Parameters, Exe_Arg_Param or Replicas"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           1,        1,                 1 , id="Set max permutation to lowest"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           6,        2,                 4 , id="Set max permutation, set replicas"),
     pytest.param(         {},        _2x2_EXE_ARG,           6,        2,                 2 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(_2x2_PARAMS,                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(         {},                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas")
))
# fmt: on
def test_step_strategy(
    # Parameterized
    params,
    exe_arg_params,
    max_perms,
    replicas,
    expected_num_jobs,
    # Other fixtures
    mock_launcher_settings,
    test_dir,
):
    jobs = Ensemble(
        "test_ensemble",
        "echo",
        ("hello", "world"),
        file_parameters=params,
        exe_arg_parameters=exe_arg_params,
        permutation_strategy="step",
        max_permutations=max_perms,
        replicas=replicas,
    ).build_jobs(mock_launcher_settings)
    assert len(jobs) == expected_num_jobs


# fmt: off
@pytest.mark.parametrize(
    "                  params,      exe_arg_params,   max_perms, replicas, expected_num_jobs",         
    (pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          30,        1,                16 , id="Set max permutation high"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,          -1,        1,                16 , id="Set max permutation negative"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           0,        1,                 1 , id="Set max permutation zero"),
     pytest.param(_2x2_PARAMS,                None,           4,        1,                 4 , id="No exe arg params or Replicas"),
     pytest.param(       None,        _2x2_EXE_ARG,           4,        1,                 4 , id="No Parameters or Replicas"),
     pytest.param(       None,                None,           4,        1,                 1 , id="No Parameters, Exe_Arg_Param or Replicas"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           1,        1,                 1 , id="Set max permutation to lowest"),
     pytest.param(_2x2_PARAMS,        _2x2_EXE_ARG,           6,        2,                12 , id="Set max permutation, set replicas"),
     pytest.param(         {},        _2x2_EXE_ARG,           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(_2x2_PARAMS,                  {},           6,        2,                 8 , id="Set params as dict, set max permutations and replicas"),
     pytest.param(         {},                  {},           6,        2,                 2 , id="Set params as dict, set max permutations and replicas")
))
# fmt: on
def test_random_strategy(
    # Parameterized
    params,
    exe_arg_params,
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
        exe_arg_parameters=exe_arg_params,
        permutation_strategy="random",
        max_permutations=max_perms,
        replicas=replicas,
    ).build_jobs(mock_launcher_settings)
    assert len(jobs) == expected_num_jobs
