import pytest

from smartsim import Experiment
from smartsim.database import (
    CobaltOrchestrator,
    Orchestrator,
    PBSOrchestrator,
    SlurmOrchestrator,
)
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError


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

    module_str = db._rai_module
    assert "THREADS_PER_QUEUE" in module_str
    assert "INTRA_OP_PARALLELISM" in module_str
    assert "INTER_OP_PARALLELISM" in module_str


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

    assert not db.is_active()

    # check if orchestrator.get_address() raises an exception
    with pytest.raises(SmartSimError):
        db.get_address()


def test_catch_local_db_errors():

    # local database with more than one node not allowed
    with pytest.raises(SSUnsupportedError):
        db = Orchestrator(db_nodes=2)

    # Run command for local orchestrator not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(run_command="srun")

    # Batch mode for local orchestrator is not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(batch=True)


#####  PBS  ######


def test_pbs_set_run_arg():
    orc = PBSOrchestrator(6780, db_nodes=3, batch=False, interface="lo")
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )
    orc.set_run_arg("pes-per-numa-node", "5")
    assert all(
        ["pes-per-numa-node" not in db.run_settings.run_args for db in orc.entities]
    )


def test_pbs_set_batch_arg():
    orc = PBSOrchestrator(6780, db_nodes=3, batch=False, interface="lo")
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = PBSOrchestrator(6780, db_nodes=3, batch=True, interface="lo")
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"
    orc2.set_batch_arg("N", "another_name")
    assert "N" not in orc2.batch_settings.batch_args


##### Slurm ######


def test_slurm_set_run_arg():
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=False, interface="lo")
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )


def test_slurm_set_batch_arg():
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=False, interface="lo")
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = SlurmOrchestrator(6780, db_nodes=3, batch=True, interface="lo")
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"


###### Cobalt ######


def test_set_run_arg():
    orc = CobaltOrchestrator(6780, db_nodes=3, batch=False, interface="lo")
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )
    orc.set_run_arg("pes-per-numa-node", "2")
    assert all(
        ["pes-per-numa-node" not in db.run_settings.run_args for db in orc.entities]
    )


def test_set_batch_arg():
    orc = CobaltOrchestrator(6780, db_nodes=3, batch=False, interface="lo")
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = CobaltOrchestrator(6780, db_nodes=3, batch=True, interface="lo")
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"
    orc2.set_batch_arg("outputprefix", "new_output/")
    assert "outputprefix" not in orc2.batch_settings.batch_args
