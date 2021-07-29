import pytest

from smartsim.database import LSFOrchestrator
from smartsim.error import SmartSimError, SSUnsupportedError

# Tests which don't require launching the orchestrator


def test_orc_converter_lsf():
    def converter(host):
        int_dict = {"host1": "HOST1-IB", "host2": "HOST2-IB"}
        if host in int_dict.keys():
            return int_dict[host]
        else:
            return ""

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
        host_map=converter,
    )
    assert orc.entities[0].hosts == ["HOST1-IB", "HOST2-IB"]

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
        host_map=None,
    )
    assert orc.entities[0].hosts == ["batch", "host1", "host2"]

    orc.set_run_arg("erf_output_name", "ERF_OUTPUT")

    assert all(
        [
            db.run_settings.run_args["erf_output_name"] == "ERF_OUTPUT"
            for db in orc.entities
        ]
    )


def test_catch_orc_errors_lsf():
    with pytest.raises(SSUnsupportedError):
        orc = LSFOrchestrator(
            6780, db_nodes=2, db_per_host=2, batch=False, hosts=["host1", "host2"]
        )

    def bad_converter(host):
        return "TWO WORDS"

    with pytest.raises(ValueError):
        orc = LSFOrchestrator(
            6780,
            db_nodes=3,
            batch=False,
            hosts=["batch", "host1", "host2"],
            host_map=bad_converter,
        )

    def bad_converter_2(host):
        return "*" * 300

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=False,
        hosts=["batch"],
        host_map=bad_converter_2,
    )

    assert ["*" * 256] == orc.entities[0]._hosts

    def bad_converter_3(host):
        # Something very stupid
        return bad_converter_2

    with pytest.raises(TypeError):
        orc = LSFOrchestrator(
            6780,
            db_nodes=3,
            batch=False,
            hosts=["batch", "host1", "host2"],
            host_map=bad_converter_3,
        )


def test_orc_converter_lsf_batch():
    def converter(host):
        int_dict = {"host1": "HOST1-IB", "host2": "HOST2-IB"}
        if host in int_dict.keys():
            return int_dict[host]
        else:
            return ""

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        host_map=converter,
    )
    assert orc.entities[0].hosts == ["HOST1-IB", "HOST2-IB"]
    assert orc.batch_settings.batch_args["m"] == '"batch host1 host2"'


def test_set_run_args():
    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        host_map=None,
    )

    orc.set_run_arg("l", "gpu-gpu")
    assert all(["l" not in db.run_settings.run_args for db in orc.entities])


def test_set_batch_args():

    orc = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
        host_map=None,
    )

    with pytest.raises(SmartSimError):
        orc.set_batch_arg("P", "MYPROJECT")

    orc2 = LSFOrchestrator(
        6780,
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        host_map=None,
    )

    orc2.set_batch_arg("D", "102400000")
    assert orc2.batch_settings.batch_args["D"] == "102400000"
