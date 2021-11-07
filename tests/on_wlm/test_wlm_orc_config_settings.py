import pytest

from smartsim import Experiment
from smartsim.error import SmartSimError

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_config_methods_on_wlm(fileutils, wlmutils):
    """test setting single node orchestrator configurations"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test_config_methods_on_wlm"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    db = wlmutils.get_orchestrator()
    db.set_path(test_dir)

    exp.start(db)
    try:
        db.enable_checkpoints(1)
        db.set_max_memory("3gb")
        db.set_eviction_strategy("allkeys-lru")
        db.set_max_clients(50_000)
        db.set_max_message_size(2_147_483_648)
    except:
        exp.stop(db)
        assert False


def test_config_methods_on_wlm_cluster(fileutils, wlmutils):
    """test setting clustered 3-node orchestrator configurations"""
    launcher = wlmutils.get_test_launcher()

    exp_name = "test_config_methods_on_wlm_cluster"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    db = wlmutils.get_orchestrator(nodes=3)
    db.set_path(test_dir)

    exp.start(db)
    try:
        db.enable_checkpoints(1)
        db.set_max_memory("3gb")
        db.set_eviction_strategy("allkeys-lru")
        db.set_max_clients(50_000)
        db.set_max_message_size(2_147_483_648)
    except:
        exp.stop(db)
        assert False


def test_config_methods_inactive_on_wlm(fileutils, wlmutils):
    """Ensure a SmartSimError is raised when trying to
    set configurations on an inactive database
    """
    launcher = wlmutils.get_test_launcher()

    exp_name = "test_config_methods_inactive_on_wlm"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    db = wlmutils.get_orchestrator()
    db.set_path(test_dir)

    exp.start(db)
    exp.stop(db)

    with pytest.raises(SmartSimError):
        db.enable_checkpoints(1)
    with pytest.raises(SmartSimError):
        db.set_max_memory("3gb")
    with pytest.raises(SmartSimError):
        db.set_eviction_strategy("allkeys-lru")
    with pytest.raises(SmartSimError):
        db.set_max_clients(50_000)
    with pytest.raises(SmartSimError):
        db.set_max_message_size(2_147_483_648)


def test_config_methods_inactive_on_wlm_cluster(fileutils, wlmutils):
    """Ensure a SmartSimError is raised when trying to
    set configurations on an inactive clustered 3-node database
    """
    launcher = wlmutils.get_test_launcher()

    exp_name = "test_config_methods_inactive_on_wlm_cluster"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    db = wlmutils.get_orchestrator(nodes=3)
    db.set_path(test_dir)

    exp.start(db)
    exp.stop(db)

    with pytest.raises(SmartSimError):
        db.enable_checkpoints(1)
    with pytest.raises(SmartSimError):
        db.set_max_memory("3gb")
    with pytest.raises(SmartSimError):
        db.set_eviction_strategy("allkeys-lru")
    with pytest.raises(SmartSimError):
        db.set_max_clients(50_000)
    with pytest.raises(SmartSimError):
        db.set_max_message_size(2_147_483_648)


def test_bad_db_conf_on_wlm(fileutils, wlmutils):
    """Ensure SmartSimErrors are raised for all kinds
    of invalid key value pairs
    """
    launcher = wlmutils.get_test_launcher()

    exp_name = "test_bad_db_conf_on_wlm"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    db = wlmutils.get_orchestrator()
    db.set_path(test_dir)

    exp.start(db)
    bad_configs = {
        "save": [
            -1,  # frequency must be positive
            2.4,  # frequency must be specified in whole seconds
        ],
        "maxmemory": [
            "29GG",  # invalid memory form
            str(2 ** 65) + "gb",  # memory is too much
            99,  # memory form must be a string
            "3.5gb",  # invalid memory form
        ],
        "maxclients": [
            -3,  # number clients must be positive
            2.9,  # number of clients must be an integer
            2 ** 65,  # number of clients is too large
        ],
        "proto-max-bulk-len": [
            100,  # max message size can't be smaller than 1mb
            101.1,  # max message size must be an integer
            "9.9gb",  # invalid memory form
        ],
        "maxmemory-policy": ["invalid-policy"],  # must use a valid maxmemory policy
        "invalid-parameter": ["99"],  # invalid key - no such configuration exists
    }

    for key, value_list in bad_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                db.set_db_conf(key, value)

    exp.stop(db)


def test_bad_db_conf_on_wlm_cluster(fileutils, wlmutils):
    """Ensure SmartSimErrors are raised for all kinds
    of invalid key value pairs
    """
    launcher = wlmutils.get_test_launcher()

    exp_name = "test_bad_db_conf_on_wlm_cluster"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    db = wlmutils.get_orchestrator(nodes=3)
    db.set_path(test_dir)

    exp.start(db)
    bad_configs = {
        "save": [
            -1,  # frequency must be positive
            2.4,  # frequency must be specified in whole seconds
        ],
        "maxmemory": [
            "29GG",  # invalid memory form
            str(2 ** 65) + "gb",  # memory is too much
            99,  # memory form must be a string
            "3.5gb",  # invalid memory form
        ],
        "maxclients": [
            -3,  # number clients must be positive
            2.9,  # number of clients must be an integer
            2 ** 65,  # number of clients is too large
        ],
        "proto-max-bulk-len": [
            100,  # max message size can't be smaller than 1mb
            101.1,  # max message size must be an integer
            "9.9gb",  # invalid memory form
        ],
        "maxmemory-policy": ["invalid-policy"],  # must use a valid maxmemory policy
        "invalid-parameter": ["99"],  # invalid key - no such configuration exists
    }

    for key, value_list in bad_configs.items():
        for value in value_list:
            with pytest.raises(SmartSimError):
                db.set_db_conf(key, value)

    exp.stop(db)
