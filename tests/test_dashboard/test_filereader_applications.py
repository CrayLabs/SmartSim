import pytest
from smartsim._core._dashboard.utils.FileReader import ManifestReader

application1 = {
    "name": "app1",
    "path": "app/1/path",
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

application3 = {
    "name": "app3",
    "path": "app/3/path",
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
    "db_models": [
        {"model1": {"backend": "model1_tf", "device": "model1_cpu"}},
        {"model2": {"backend": "model2_tf", "device": "model2_cpu"}},
    ],
    "job_id": "111",
    "step_id": 111,
}

parameterize_creator = pytest.mark.parametrize(
    "json_file, app_name, application",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app1",
            application1,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app3",
            application3,
        ),
        pytest.param("file_doesnt_exist.json", "app4", None),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app_doesnt_exist",
            None,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_apps_manifest.json", "app1", None
        ),
    ],
)


@parameterize_creator
def test_get_entity_apps(json_file, app_name, application):
    dash_data = ManifestReader(json_file)
    app = dash_data.get_entity(app_name, dash_data.applications)
    assert app == application


parameterize_creator = pytest.mark.parametrize(
    "json_file, app_name, key, expected_key_value",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app1",
            "path",
            "app/1/path",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app2",
            "path",
            "app/2/path",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app1",
            "batch_settings",
            {"batch_cmd": "command", "batch_args": {"arg1": "string1", "arg2": None}},
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app2",
            "exe_args",
            ["string"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            "exe_args",
            [],
        ),
        pytest.param("file_doesnt_exist.json", "app1", "path", ""),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app_doesnt_exist",
            "path",
            "",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_app_manifest.json",
            "app1",
            "path",
            "",
        ),
    ],
)


@parameterize_creator
def test_get_entity_value_apps(json_file, app_name, key, expected_key_value):
    dash_data = ManifestReader(json_file)
    app = dash_data.get_entity(app_name, dash_data.applications)
    app_key_value = dash_data.get_entity_value(key, app)
    assert app_key_value == expected_key_value


parameterize_creator = pytest.mark.parametrize(
    "json_file, app_name, dict_name, keys, values",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app1",
            "batch_settings",
            ["batch_cmd", "arg1", "arg2"],
            ["command", "string1", "None"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app2",
            "files",
            ["Symlink", "Symlink", "Configure", "Copy", "Copy"],
            ["file1", "file2", "file3", "file4", "file5"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app1",
            "run_settings",
            ["exe", "run_command", "arg1", "arg2"],
            ["echo", "srun", "string1", "None"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            "run_settings",
            [],
            [],
        ),
        pytest.param("file_doesnt_exist.json", "app2", "run_settings", [], []),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_apps_manifest.json",
            "app1",
            "run_settings",
            [],
            [],
        ),
    ],
)


@parameterize_creator
def test_get_entity_dict_keys_and_values_apps(
    json_file, app_name, dict_name, keys, values
):
    dash_data = ManifestReader(json_file)
    app = dash_data.get_entity(app_name, dash_data.applications)
    k, v = dash_data.get_entity_dict_keys_and_values(dict_name, app)
    assert k == keys
    assert v == values


parameterize_creator = pytest.mark.parametrize(
    "json_file, app_name, values",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "app1",
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
            ],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_apps_manifest.json",
            "app1",
            {"Name": [], "Type": [], "Backend": [], "Device": []},
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            {"Name": [], "Type": [], "Backend": [], "Device": []},
        ),
    ],
)


@parameterize_creator
def test_get_loaded_entities_apps(json_file, app_name, values):
    dash_data = ManifestReader(json_file)
    app = dash_data.get_entity(app_name, dash_data.applications)
    entities = dash_data.get_loaded_entities(app)
    assert entities == values
