import pytest

from smartsim import Experiment
from smartsim.entity import Model, Ensemble
from smartsim.error import EntityExistsError, SSUnsupportedError
from smartsim.settings import RunSettings, SbatchSettings, SrunSettings
from smartsim.settings.mpirunSettings import _OpenMPISettings
from smartsim._core.launcher.step import SrunStep, SbatchStep

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
    rs = _OpenMPISettings("python", exe_args="sleep.py")

    # make it an mpmd model
    rs_2 = _OpenMPISettings("python", exe_args="sleep.py")
    rs.make_mpmd(rs_2)

    model = exp.create_model("bad_colo_model", rs)

    # make it co-located which should raise and error
    with pytest.raises(SSUnsupportedError):
        model.colocate_db()


def test_attach_batch_settings_to_model():
    exp = Experiment("experiment", launcher="slurm")
    bs = SbatchSettings()
    rs = SrunSettings("python", exe_args="sleep.py")

    model_wo_bs = exp.create_model("test_model", run_settings=rs)
    assert model_wo_bs.batch_settings is None

    model_w_bs = exp.create_model("test_model_2", run_settings=rs, batch_settings=bs)
    assert isinstance(model_w_bs.batch_settings, SbatchSettings)


@pytest.fixture
def monkeypatch_exp_controller(monkeypatch):
    def _monkeypatch_exp_controller(exp):
        entity_steps = []

        def start_wo_job_manager(self, manifest, block=True, kill_on_interrupt=True):
            self._launch(manifest)

        def launch_step_nop(self, step, entity):
            entity_steps.append((step, entity))

        monkeypatch.setattr(exp._control, "start", start_wo_job_manager.__get__(exp._control, type(exp._control))) 
        monkeypatch.setattr(exp._control, "_launch_step", launch_step_nop.__get__(exp._control, type(exp._control)))

        return entity_steps
    return _monkeypatch_exp_controller


def test_model_with_batch_settings_makes_batch_step(monkeypatch_exp_controller):
    exp = Experiment("experiment", launcher="slurm")
    bs = SbatchSettings()
    rs = SrunSettings("python", exe_args="sleep.py")
    model = exp.create_model("test_model", run_settings=rs, batch_settings=bs)

    entity_steps = monkeypatch_exp_controller(exp)
    exp.start(model)

    assert len(entity_steps) == 1
    step, entity = entity_steps[0]
    assert isinstance(entity, Model)
    assert isinstance(step, SbatchStep)

def test_model_without_batch_settings_makes_run_step(monkeypatch_exp_controller):
    exp = Experiment("experiment", launcher="slurm")
    rs = SrunSettings("python", exe_args="sleep.py")
    model = exp.create_model("test_model", run_settings=rs)

    entity_steps = monkeypatch_exp_controller(exp)
    exp.start(model)

    assert len(entity_steps) == 1
    step, entity = entity_steps[0]
    assert isinstance(entity, Model)
    assert isinstance(step, SrunStep)

def test_models_batch_settings_are_ignored_in_ensemble(monkeypatch_exp_controller):
    exp = Experiment("experiment", launcher="slurm")
    bs_1 = SbatchSettings(nodes=5)
    rs = SrunSettings("python", exe_args="sleep.py")
    model = exp.create_model("test_model", run_settings=rs, batch_settings=bs_1)

    bs_2 = SbatchSettings(nodes=10)
    ens = exp.create_ensemble("test_ensemble", batch_settings=bs_2)
    ens.add_model(model)

    entity_steps = monkeypatch_exp_controller(exp)
    exp.start(ens)

    assert len(entity_steps) == 1
    step, entity = entity_steps[0]
    assert isinstance(entity, Ensemble)
    assert isinstance(step, SbatchStep)
    assert step.batch_settings.batch_args["nodes"] == 10
    assert len(step.step_cmds) == 1
    step_cmd = step.step_cmds[0]
    assert any("srun" in tok for tok in step_cmd)  # call the model using run settings
    assert not any("sbatch" in tok for tok in step_cmd)  # no sbatch in sbatch


