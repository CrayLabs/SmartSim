import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError


def test_orc_parameters():
    threads_per_queue = 2
    inter_op_threads = 2
    intra_op_threads = 2
    db = Orchestrator(
        db_nodes=1,
        threads_per_queue=threads_per_queue,
        inter_op_threads=inter_op_threads,
        intra_op_threads=intra_op_threads,
    )
    assert db.queue_threads == threads_per_queue
    assert db.inter_threads == inter_op_threads
    assert db.intra_threads == intra_op_threads

    module_str = db._get_AI_module()
    assert "THREADS_PER_QUEUE" in module_str
    assert "INTRA_OP_THREADS" in module_str
    assert "INTER_OP_THREADS" in module_str


def test_is_not_active():
    db = Orchestrator(db_nodes=1)
    assert not db.is_active()


def test_inactive_orc_get_address():
    db = Orchestrator()
    with pytest.raises(SmartSimError):
        db.get_address()


def test_orc_active_functions(fileutils):
    exp_name = "test_orc_active_functions"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir(exp_name)

    db = Orchestrator(port=6780)
    db.set_path(test_dir)

    exp.start(db)

    # check if the orchestrator is active
    assert db.is_active()

    # check if the orchestrator can get the address
    assert db.get_address() == ["127.0.0.1:6780"]

    exp.stop(db)

    # TODO: Update is_active code after smartredis 0.2.0 is released
    # assert not db.is_active()

    # check if orchestrator.get_addree() raises an exception
    with pytest.raises(SmartSimError):
        db.get_address()


def test_catch_local_db_errors():

    # local database with more than one node not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(db_nodes=2)

    # MPMD local orchestrator not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(dpn=2)
