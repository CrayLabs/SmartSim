import pytest

from smartsim import Experiment
from smartsim.database import (
    CobaltOrchestrator,
    Orchestrator,
    PBSOrchestrator,
    SlurmOrchestrator,
)
from smartsim.error import SmartSimError


def test_db_conf(fileutils):
    exp_name = "test_db_conf"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
    db.set_path(test_dir)

    exp.start(db)
    kv = {
        "dbfilename": "new_dump.rdb",
        "replica-priority": "99",
        "acllog-max-len": "256",
        "maxmemory": "3gb",
        "maxmemory-policy": "volatile-lru",
        "maxclients": "5000",
        "proto-max-bulk-len": "1gb",
    }
    for key, value in kv.items():
        db.set_db_conf(key, value)
    exp.stop(db)


def test_db_conf_bad_kv(fileutils):
    exp_name = "test_db_conf_bad_kv"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
    db.set_path(test_dir)

    exp.start(db)
    kv = {
        "maxmemory": "-1029-invalid",
        "invalid-parameter": "99",
        "maxmemory": 99
    }
    for key, value in kv.items():
        with pytest.raises(SmartSimError):
            db.set_db_conf(key, value)
    exp.stop(db)


def test_max_memory(fileutils):
    """Ensure setting max memory on an active
    database with a valid memory value works as expected
    """
    exp_name = "test_max_memory"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
    db.set_path(test_dir)

    exp.start(db)
    db.set_max_memory("128mb")
    exp.stop(db)


def test_max_memory_bad_mem_val(fileutils):
    """Ensure a SmartSimError is raised when
    an invalid memory value is used for setting
    the database's max memory
    """
    exp_name = "test_max_memory_bad_mem_val"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
    db.set_path(test_dir)

    exp.start(db)
    with pytest.raises(SmartSimError):
        db.set_max_memory("3_INVALID_UNIT")
    exp.stop(db)


def test_max_memory_inactive_basic(fileutils):
    """Ensure a SmartSimError is raised when trying to
    set the max memory on an inactive database
    """
    db = Orchestrator(db_nodes=1)
    with pytest.raises(SmartSimError):
        db.set_max_memory("128mb")


def test_max_memory_inactive(fileutils):
    """Ensure a SmartSimError is raised when trying to
    set the max memory on a database that was active
    in the past, but is currently not active
    """
    exp_name = "test_max_memory_inactive"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(db_nodes=1)
    db.set_path(test_dir)

    exp.start(db)
    exp.stop(db)
    with pytest.raises(SmartSimError):
        db.set_max_memory("128mb")
