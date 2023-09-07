import pytest
from smartsim._core._dashboard.utils.FileReader import ManifestReader


ensemble = {
    "name": "ensemble_3",
    "perm_strat": "string2",
    "batch_settings": {"string": "Any"},
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

ensemble2 = {
    "name": "ensemble_2",
    "perm_strat": "all-perm",
    "batch_settings": {"string": "Any1"},
    "params": {"string": ["Any1", "Any2", "Any3"]},
    "members": [
        {
            "name": "ensemble_2_member_1",
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
            "db_scripts": [],
            "db_models": [],
            "job_id": "111",
            "step_id": 111,
        }
    ],
}

parameterize_creator = pytest.mark.parametrize(
    "json_file, member_name, ensemble, key, expected_key_value",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            "path",
            "member 1 path",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            "files",
            {
                "Symlink": ["file1", "file2"],
                "Configure": ["file3"],
                "Copy": ["file4", "file5"],
            },
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            "batch_settings",
            {"batch_cmd": "command", "batch_args": {"arg1": "string1", "arg2": None}},
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_2",
            ensemble,
            "exe_args",
            ["string"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_2",
            ensemble,
            "path",
            "member 2 path",
        ),
        pytest.param("file_doesnt_exist.json", "ensemble_3_member_2", {}, "path", ""),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "",
            ensemble,
            "path",
            "",
        ),
    ],
)


@parameterize_creator
def test_get_entity_value_member(
    json_file, member_name, ensemble, key, expected_key_value
):
    dash_data = ManifestReader(json_file)
    member = dash_data.get_member(member_name, ensemble)
    member_key_value = dash_data.get_entity_value(key, member)
    assert member_key_value == expected_key_value


parameterize_creator = pytest.mark.parametrize(
    "json_file, ensemble_name, num_members, member_names",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_1",
            1,
            ["ensemble_1_member_1"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_2",
            1,
            ["ensemble_2_member_1"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3",
            2,
            ["ensemble_3_member_1", "ensemble_3_member_2"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble doesn't exist",
            0,
            [],
        ),
        pytest.param("file_doesnt_exist.json", "ensemble_3", 0, []),
    ],
)


@parameterize_creator
def test_get_ensemble_members(json_file, ensemble_name, num_members, member_names):
    dash_data = ManifestReader(json_file)
    ensemble = dash_data.get_entity(ensemble_name, dash_data.ensembles)
    members = dash_data.get_ensemble_members(ensemble)
    assert len(members) == num_members
    names = [member["name"] for member in members]
    assert names == member_names


parameterize_creator = pytest.mark.parametrize(
    "json_file, member_name, ensemble, member",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            ensemble["members"][0],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_2",
            ensemble,
            ensemble["members"][1],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "member_doesnt_exist",
            ensemble,
            None,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_ensemble_manifest.json",
            "member_doesnt_exist",
            ensemble,
            None,
        ),
        pytest.param("file_doesnt_exist.json", "member_doesnt_exist", ensemble, None),
    ],
)


@parameterize_creator
def test_get_member(json_file, member_name, ensemble, member):
    dash_data = ManifestReader(json_file)
    mem = dash_data.get_member(member_name, ensemble)
    assert mem == member


parameterize_creator = pytest.mark.parametrize(
    "json_file, member_name, ensemble, dict_name, keys, values",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            "batch_settings",
            ["batch_cmd", "arg1", "arg2"],
            ["command", "string1", "None"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_2",
            ensemble,
            "batch_settings",
            ["batch_cmd", "arg1"],
            ["command", "string1"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            "files",
            ["Symlink", "Symlink", "Configure", "Copy", "Copy"],
            ["file1", "file2", "file3", "file4", "file5"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            ensemble,
            "run_settings",
            [],
            [],
        ),
        pytest.param(
            "file_doesnt_exist.json", "ensemble_3_member_2", {}, "run_settings", [], []
        ),
    ],
)


@parameterize_creator
def test_get_entity_dict_keys_and_values_member(
    json_file, member_name, ensemble, dict_name, keys, values
):
    dash_data = ManifestReader(json_file)
    member = dash_data.get_member(member_name, ensemble)
    k, v = dash_data.get_entity_dict_keys_and_values(dict_name, member)
    assert k == keys
    assert v == values


parameterize_creator = pytest.mark.parametrize(
    "json_file, member_name, ensemble, values",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "ensemble_3_member_1",
            ensemble,
            [
                {
                    "Name": "model1",
                    "Type": "DB Model",
                    "Backend": "model1_tf",
                    "Device": "model1_cpu",
                },
                {
                    "Name": "model2",
                    "Type": "DB Model",
                    "Backend": "model2_tf",
                    "Device": "model2_cpu",
                },
                {
                    "Name": "script1",
                    "Type": "DB Script",
                    "Backend": "script1_torch",
                    "Device": "script1_cpu",
                },
                {
                    "Name": "script2",
                    "Type": "DB Script",
                    "Backend": "script2_torch",
                    "Device": "script2_gpu",
                },
            ],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app3",
            ensemble2,
            {
                "Name": [],
                "Type": [],
                "Backend": [],
                "Device": [],
            },
        ),
    ],
)


@parameterize_creator
def test_get_loaded_entities_member(json_file, member_name, ensemble, values):
    dash_data = ManifestReader(json_file)
    member = dash_data.get_member(member_name, ensemble)
    entities = dash_data.get_loaded_entities(member)
    assert entities == values
