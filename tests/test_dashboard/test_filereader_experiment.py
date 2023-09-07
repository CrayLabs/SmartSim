import pytest
from smartsim._core._dashboard.utils.FileReader import ManifestReader


parameterize_creator = pytest.mark.parametrize(
    "json_file, key, expected_key_value",
    [
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "name",
            "my-experiment",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json",
            "path",
            "experiment/path",
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json", "launcher", "local"
        ),
        pytest.param(
            "tests/test_dashboard/manifest_files/manifesttest.json", "not_a_field", ""
        ),
        pytest.param("file_doesnt_exist.json", "name", ""),
        pytest.param(
            "tests/test_dashboard/manifest_files/no_experiment_manifest.json",
            "name",
            "",
        ),
    ],
)


@parameterize_creator
def test_get_entity_value_experiment(json_file, key, expected_key_value):
    dash_data = ManifestReader(json_file)
    exp = dash_data.experiment
    exp_key_val = dash_data.get_entity_value(key, exp)
    assert exp_key_val == expected_key_value
