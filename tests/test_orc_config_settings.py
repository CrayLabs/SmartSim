import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError

try:
    from smartredis import Client

    config_setter = Client.config_set
except AttributeError:
    pytestmark = pytest.mark.skip(reason="SmartRedis version is < 0.3.0")


def test_config_methods(fileutils):
    exp_name = "test_config_methods"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
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

    exp.stop(db)


def test_config_methods_inactive(fileutils):
    """Ensure a SmartSimError is raised when trying to
    set configurations on an inactive database
    """
    exp_name = "test_config_methods_inactive"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
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


def test_bad_db_conf(fileutils):
    """Ensure SmartSimErrors are raised for all kinds
    of invalid key value pairs
    """
    exp_name = "test_bad_db_conf"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
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
