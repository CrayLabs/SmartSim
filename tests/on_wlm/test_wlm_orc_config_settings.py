import pytest

from smartsim.error import SmartSimError

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

try:
    from smartredis import Client

    config_setter = Client.config_set
except AttributeError:
    pytestmark = pytest.mark.skip(reason="SmartRedis version is < 0.3.0")


def test_config_methods_on_wlm_cluster(dbutils, db_cluster):
    """test setting clustered 3-node orchestrator configurations"""
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(db_cluster, setting)
        config_set_method(value)


def test_config_methods_inactive_on_wlm_cluster(wlmutils, dbutils):
    """Ensure a SmartSimError is raised when trying to
    set configurations on an inactive clustered 3-node database
    """
    db = wlmutils.get_orchestrator(nodes=3)
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(db, setting)
        with pytest.raises(SmartSimError):
            config_set_method(value)


def test_bad_db_conf_on_wlm_cluster(dbutils, db_cluster):
    """Ensure SmartSimErrors are raised for all kinds
    of invalid key value pairs
    """
    bad_configs = dbutils.get_bad_db_configs()
    for key, value_list in bad_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                db_cluster.set_db_conf(key, value)
