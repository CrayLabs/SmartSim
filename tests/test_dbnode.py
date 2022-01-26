import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error.errors import SmartSimError


def test_parse_db_host_error():
    orc = Orchestrator()
    orc.entities[0].path = "not/a/path"
    # Fail to obtain database hostname
    with pytest.raises(SmartSimError):
        orc.entities[0].host


def test_hosts(fileutils):
    exp_name = "test_hosts"
    exp = Experiment(exp_name)
    test_dir = fileutils.make_test_dir(exp_name)

    orc = Orchestrator(port=6888, interface="lo", launcher="local")
    orc.set_path(test_dir)
    exp.start(orc)

    thrown = False
    hosts = []
    try:
        hosts = orc.hosts
    except SmartSimError:
        thrown = True
    finally:
        # stop the database even if there is an error raised
        exp.stop(orc)
        orc.remove_stale_files()
        assert not thrown
        assert hosts == orc.hosts


def test_set_host():
    orc = Orchestrator()
    orc.entities[0].set_host("host")
    assert orc.entities[0]._host == "host"
