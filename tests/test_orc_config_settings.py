import pytest

from smartsim.error import SmartSimError

try:
    from smartredis import Client

    config_setter = Client.config_set
except AttributeError:
    pytestmark = pytest.mark.skip(reason="SmartRedis version is < 0.3.0")


def test_config_methods(dbutils, local_db):
    """Test all configuration file edit methods on an active db"""

    # test the happy path and ensure all configuration file edit methods
    # successfully execute when given correct key-value pairs
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(local_db, setting)
        config_set_method(value)

    # ensure SmartSimError is raised when Orchestrator.set_db_conf
    # is given invalid CONFIG key-value pairs
    ss_error_configs = dbutils.get_smartsim_error_db_configs()
    for key, value_list in ss_error_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                local_db.set_db_conf(key, value)

    # ensure TypeError is raised when Orchestrator.set_db_conf
    # is given either a key or a value that is not a string
    type_error_configs = dbutils.get_type_error_db_configs()
    for key, value_list in type_error_configs.items():
        for value in value_list:
            with pytest.raises(TypeError):
                local_db.set_db_conf(key, value)


def test_config_methods_inactive(wlmutils, dbutils):
    """Ensure a SmartSimError is raised when trying to
    set configurations on an inactive database
    """
    db = wlmutils.get_orchestrator()
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(db, setting)
        with pytest.raises(SmartSimError):
            config_set_method(value)
