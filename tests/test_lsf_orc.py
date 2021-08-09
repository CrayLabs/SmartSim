import pytest

from smartsim.database import LSFOrchestrator
from smartsim.error import SmartSimError, SSUnsupportedError

# Tests which don't require launching the orchestrator


def test_catch_orc_errors_lsf():
    with pytest.raises(SSUnsupportedError):
        orc = LSFOrchestrator(
            6780, db_nodes=2, db_per_host=2, batch=False
        )

def test_set_run_args():

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
    )
    orc.set_run_arg("l", "gpu-gpu")
    assert all(["l" not in db.run_settings.run_args for db in orc.entities])

def test_set_batch_args():

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
    )
    assert orc.batch_settings.batch_args["m"] == '"batch host1 host2"'

    with pytest.raises(SmartSimError):
        orc.set_batch_arg("P", "MYPROJECT")

    orc2 = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
    )

    orc2.set_batch_arg("D", "102400000")
    assert orc2.batch_settings.batch_args["D"] == "102400000"
