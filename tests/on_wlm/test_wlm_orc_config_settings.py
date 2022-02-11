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


def test_config_methods_on_wlm_cluster(dbutils, db):
    """Test all configuration file edit methods on single node WLM db"""

    # test the happy path and ensure all configuration file edit methods
    # successfully execute when given correct key-value pairs
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(db, setting)
        config_set_method(value)

    # ensure SmartSimError is raised when a clustered database's
    # Orchestrator.set_db_conf is given invalid CONFIG key-value pairs
    ss_error_configs = dbutils.get_smartsim_error_db_configs()
    for key, value_list in ss_error_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                db.set_db_conf(key, value)

    # ensure TypeError is raised when a clustered database's
    # Orchestrator.set_db_conf is given invalid CONFIG key-value pairs
    type_error_configs = dbutils.get_type_error_db_configs()
    for key, value_list in type_error_configs.items():
        for value in value_list:
            with pytest.raises(TypeError):
                db.set_db_conf(key, value)


def test_config_methods_on_wlm_cluster(dbutils, db_cluster):
    """Test all configuration file edit methods on an active clustered db"""

    # test the happy path and ensure all configuration file edit methods
    # successfully execute when given correct key-value pairs
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(db_cluster, setting)
        config_set_method(value)

    # ensure SmartSimError is raised when a clustered database's
    # Orchestrator.set_db_conf is given invalid CONFIG key-value pairs
    ss_error_configs = dbutils.get_smartsim_error_db_configs()
    for key, value_list in ss_error_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                db_cluster.set_db_conf(key, value)

    # ensure TypeError is raised when a clustered database's
    # Orchestrator.set_db_conf is given invalid CONFIG key-value pairs
    type_error_configs = dbutils.get_type_error_db_configs()
    for key, value_list in type_error_configs.items():
        for value in value_list:
            with pytest.raises(TypeError):
                db_cluster.set_db_conf(key, value)
