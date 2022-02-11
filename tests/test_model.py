import pytest

from smartsim import Experiment
from smartsim.error import EntityExistsError, SSUnsupportedError
from smartsim.settings import RunSettings, MpirunSettings


def test_register_incoming_entity_preexists():
    exp = Experiment("experiment", launcher="local")
    rs = RunSettings("python", exe_args="sleep.py")
    ensemble = exp.create_ensemble(name="ensemble", replicas=1, run_settings=rs)
    m = exp.create_model("model", run_settings=rs)
    m.register_incoming_entity(ensemble["ensemble_0"])
    assert len(m.incoming_entities) == 1
    with pytest.raises(EntityExistsError):
        m.register_incoming_entity(ensemble["ensemble_0"])


def test_disable_key_prefixing():
    exp = Experiment("experiment", launcher="local")
    rs = RunSettings("python", exe_args="sleep.py")
    m = exp.create_model("model", run_settings=rs)
    m.disable_key_prefixing()
    assert m.query_key_prefixing() == False

def test_catch_colo_mpmd_model():
    exp = Experiment("experiment", launcher="local")
    rs = MpirunSettings("python", exe_args="sleep.py")

    # make it an mpmd model
    rs_2 = MpirunSettings("python", exe_args="sleep.py")
    rs.make_mpmd(rs_2)

    model = exp.create_model("bad_colo_model", rs)

    # make it co-located which should raise and error
    with pytest.raises(SSUnsupportedError):
        model.colocate_db()
