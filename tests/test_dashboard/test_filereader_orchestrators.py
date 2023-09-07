import pytest
from smartsim._core._dashboard.utils.FileReader import ManifestReader

orch1 = {
    "name": "orchestrator_1",
    "launcher": "local",
    "port": "12345",
    "interface": "lo",
    "db_hosts": ["string"],
}

orch2 = {
    "name": "orchestrator_2",
    "launcher": "local",
    "port": 22222,
    "interface": ["lo"],
    "db_hosts": ["string"],
}


parameterize_creator = pytest.mark.parametrize(
    "json_file, orc_name, orchestrator",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_1",
            orch1,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_2",
            orch2,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orc_doesnt_exist",
            None,
        ),
        pytest.param("file_doesnt_exist.json", "orchestrator_1", None),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_orchestrator_manifest.json",
            "orchestrator_1",
            None,
        ),
    ],
)


@parameterize_creator
def test_get_entity_orchestrator(json_file, orc_name, orchestrator):
    dash_data = ManifestReader(json_file)
    orc = dash_data.get_entity(orc_name, dash_data.orchestrators)
    assert orc == orchestrator


parameterize_creator = pytest.mark.parametrize(
    "json_file, orc_name, key, expected_key_value",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_1",
            "port",
            "12345",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_2",
            "port",
            22222,
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_1",
            "interface",
            "lo",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_0",
            "interface",
            "lo, lo2",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_3",
            "launcher",
            "local",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "orchestrator_3",
            "db_hosts",
            ["host1", "host2", "host3"],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            "db_hosts",
            [],
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            "port",
            "",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "doesnt_exist",
            "interface",
            "",
        ),
        pytest.param("file_doesnt_exist.json", "orchestrator_1", "launcher", ""),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_orchestrator_manifest.json",
            "orchestrator_1",
            "launcher",
            "",
        ),
    ],
)


@parameterize_creator
def test_get_entity_value_orchestrator(json_file, orc_name, key, expected_key_value):
    dash_data = ManifestReader(json_file)
    orc = dash_data.get_entity(orc_name, dash_data.orchestrators)
    orc_key_value = dash_data.get_entity_value(key, orc)
    assert orc_key_value == expected_key_value
