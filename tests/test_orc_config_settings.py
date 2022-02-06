import pytest

from smartsim.error import SmartSimError

try:
    from smartredis import Client

    config_setter = Client.config_set
except AttributeError:
    pytestmark = pytest.mark.skip(reason="SmartRedis version is < 0.3.0")


def test_config_methods(dbutils, db):
    """Ensure all configuration file edit methods in the Orchestrator pass"""
    configs = dbutils.get_db_configs()
    for setting, value in configs.items():
        config_set_method = dbutils.get_config_edit_method(db, setting)
        config_set_method(value)


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


def test_bad_db_conf(dbutils, db):
    """Ensure SmartSimErrors are raised for all kinds
    of invalid key value pairs
    """
    bad_configs = dbutils.get_bad_db_configs()
    for key, value_list in bad_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                db.set_db_conf(key, value)
