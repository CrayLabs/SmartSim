import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error.errors import SmartSimError


def test_parse_db_host_error():
    orc = Orchestrator()
    # Fail to obtain database hostname
    with pytest.raises(SmartSimError):
        orc.entities[0].host


def test_parse_db_host():
    # check that an Error is NOT raised
    exp = Experiment("test_dbnode")
    orc = Orchestrator()
    exp.generate(orc)
    exp.start(orc)
    orc.entities[0]._parse_db_host()
    exp.stop(orc)
    orc.remove_stale_files()


def test_set_host():
    orc = Orchestrator()
    orc.entities[0].set_host("host")
    assert orc.entities[0]._host == "host"
