import pytest

from smartsim._core.control.controller import Controller
from smartsim.settings.slurmSettings import SbatchSettings, SrunSettings
from smartsim._core.launcher.step import Step
from smartsim.entity.ensemble import Ensemble
from smartsim.database.orchestrator import Orchestrator

controller = Controller()

rs = SrunSettings('echo', ['spam', 'eggs'])
bs = SbatchSettings()

ens = Ensemble("ens", params={}, run_settings=rs, batch_settings=bs, replicas=3)
orc = Orchestrator(db_nodes=3, batch=True, launcher="slurm", run_command="srun")

class MockStep(Step):
    @staticmethod
    def _create_unique_name(name):
        return name

    def add_to_batch(self, step):
        ...

    def get_launch_cmd(self):
        return []

@pytest.mark.parametrize("collection", [
    pytest.param(ens, id="Ensemble"),
    pytest.param(orc, id="Database"),
])
def test_controller_batch_step_creation_preserves_entity_order(collection, monkeypatch):
    monkeypatch.setattr(controller._launcher, "create_step",
                        lambda name, path, settings: MockStep(name, path, settings))
    entity_names = [x.name for x in collection.entities]
    assert len(entity_names) == len(set(entity_names))
    _, steps = controller._create_batch_job_step(collection)
    assert entity_names == [step.name for step in steps]

    
