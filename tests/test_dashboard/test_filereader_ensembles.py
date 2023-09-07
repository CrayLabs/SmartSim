import pytest
from smartsim._core._dashboard.utils.FileReader import ManifestReader

ensemble1 = {
    "name": "ensemble_1",
    "perm_strat": "string0",
    "batch_settings": {"string": "Any0"},
    "params": {"string": ["Any1", "Any3"]},
    "members": [
        {
            "name": "ensemble_1_member_1",
            "path": "string",
            "exe_args": ["string"],
            "batch_settings": {
                "batch_cmd": "command",
                "batch_args": {"arg1": "string1", "arg2": None},
            },
            "run_settings": {
                "exe": "echo",
                "run_command": "srun",
                "run_args": {"arg1": "string1", "arg2": None},
            },
            "params": {"string": "Any"},
            "files": {
                "Symlink": ["file1", "file2"],
                "Configure": ["file3"],
                "Copy": ["file4", "file5"],
            },
            "colocated_db_settings": {
                "protocol": "TCP/IP",
                "port": 1111,
                "interface": "lo",
                "db_cpus": 1,
                "limit_app_cpus": "True",
                "debug": "False",
            },
            "db_scripts": [
                {"script1": {"backend": "script1_torch", "device": "script1_cpu"}},
                {"script2": {"backend": "script2_torch", "device": "script2_gpu"}},
            ],
            "db_models": [
                {"model1": {"backend": "model1_tf", "device": "model1_cpu"}},
                {"model2": {"backend": "model2_tf", "device": "model2_cpu"}},
            ],
            "job_id": "111",
            "step_id": 111,
        }
    ],
}

ensemble3 = {
    "name": "ensemble_3",
    "perm_strat": "string2",
    "batch_settings": {"string": "Any1"},
    "params": {"string": ["Any1", "Any2", "Any3"]},
    "members": [
        {
            "name": "ensemble_3_member_1",
            "path": "member 1 path",
            "exe_args": ["string"],
            "batch_settings": {
                "batch_cmd": "command",
                "batch_args": {"arg1": "string1", "arg2": None},
            },
            "run_settings": {
                "exe": "echo",
                "run_command": "srun",
                "run_args": {"arg1": "string1", "arg2": None},
            },
            "params": {"string": "Any"},
            "files": {
                "Symlink": ["file1", "file2"],
                "Configure": ["file3"],
                "Copy": ["file4", "file5"],
            },
            "colocated_db_settings": {
                "protocol": "TCP/IP",
                "port": 1111,
                "interface": "lo",
                "db_cpus": 1,
                "limit_app_cpus": "True",
                "debug": "False",
            },
            "db_scripts": [
                {"script1": {"backend": "script1_torch", "device": "script1_cpu"}},
                {"script2": {"backend": "script2_torch", "device": "script2_gpu"}},
            ],
            "db_models": [
                {"model1": {"backend": "model1_tf", "device": "model1_cpu"}},
                {"model2": {"backend": "model2_tf", "device": "model2_cpu"}},
            ],
            "job_id": "111",
            "step_id": 111,
        },
        {
            "name": "ensemble_3_member_2",
            "path": "member 2 path",
            "exe_args": ["string"],
            "batch_settings": {
                "batch_cmd": "command",
                "batch_args": {"arg1": "string1"},
            },
            "run_settings": {
                "exe": "echo",
                "run_command": "srun",
                "run_args": {"arg1": "string1", "arg2": None},
            },
            "params": {"string": "Any"},
            "files": {
                "Symlink": ["file1", "file2"],
                "Configure": ["file3"],
                "Copy": ["file4", "file5"],
            },
            "colocated_db_settings": {
                "protocol": "TCP/IP",
                "port": 1111,
                "interface": "lo",
                "db_cpus": 1,
                "limit_app_cpus": "True",
                "debug": "False",
            },
            "db_scripts": [
                {"script1": {"backend": "script1_torch", "device": "script1_cpu"}},
                {"script2": {"backend": "script2_torch", "device": "script2_gpu"}},
            ],
            "db_models": [
                {"model1": {"backend": "model1_tf", "device": "model1_cpu"}},
                {"model2": {"backend": "model2_tf", "device": "model2_cpu"}},
            ],
            "job_id": "111",
            "step_id": 111,
        },
    ],
}

parameterize_creator = pytest.mark.parametrize(
    "json_file, ensemble_name, ensemble",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_1",
            ensemble1,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3",
            ensemble3,
        ),
        pytest.param("file_doesnt_exist.json", "ensemble4", None),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_doesnt_exist",
            None,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_ensembles_manifest.json",
            "ensemble_1",
            None,
        ),
    ],
)


@parameterize_creator
def test_get_entity_ensemble(json_file, ensemble_name, ensemble):
    dash_data = ManifestReader(json_file)
    ens = dash_data.get_entity(ensemble_name, dash_data.ensembles)
    assert ens == ensemble


parameterize_creator = pytest.mark.parametrize(
    "json_file, ensemble_name, key, expected_key_value",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_2",
            "perm_strat",
            "all-perm",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_2",
            "params",
            {"string": ["Any1", "Any2", "Any3"]},
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_1",
            "batch_settings",
            {"string": "Any0"},
        ),
        pytest.param("file_doesnt_exist.json", "ensemble_2", "perm_strategy", ""),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_doesnt_exist",
            "perm_strat",
            "",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_ensembles_manifest.json",
            "ensemble_1",
            "perm_strat",
            "",
        ),
    ],
)


@parameterize_creator
def test_get_entity_value_ensemble(json_file, ensemble_name, key, expected_key_value):
    dash_data = ManifestReader(json_file)
    ens = dash_data.get_entity(ensemble_name, dash_data.ensembles)
    ens_key_value = dash_data.get_entity_value(key, ens)
    assert ens_key_value == expected_key_value


parameterize_creator = pytest.mark.parametrize(
    "json_file, ensemble_name, dict_name, keys, values",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_1",
            "batch_settings",
            ["string"],
            ["Any0"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_2",
            "params",
            ["string"],
            ["Any1, Any2, Any3"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_1",
            "params",
            ["string"],
            ["Any1, Any3"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            "batch_settings",
            [],
            [],
        ),
        pytest.param("file_doesnt_exist.json", "ensemble_1", "parameters", [], []),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_ensembles_manifest.json",
            "ensemble_1",
            "params",
            [],
            [],
        ),
    ],
)


@parameterize_creator
def test_get_entity_dict_keys_and_values_ensemble(
    json_file, ensemble_name, dict_name, keys, values
):
    dash_data = ManifestReader(json_file)
    ens = dash_data.get_entity(ensemble_name, dash_data.ensembles)
    k, v = dash_data.get_entity_dict_keys_and_values(dict_name, ens)
    assert k == keys
    assert v == values
