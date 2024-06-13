import pytest

from smartsim._core.launcher.step import (
    AprunStep,
    BsubBatchStep,
    JsrunStep,
    LocalStep,
    MpiexecStep,
    MpirunStep,
    OrterunStep,
    QsubBatchStep,
    SbatchStep,
    SrunStep,
)
from smartsim.entity import Model
from smartsim.settings import (
    AprunSettings,
    BsubBatchSettings,
    JsrunSettings,
    MpirunSettings,
    OrterunSettings,
    QsubBatchSettings,
    RunSettings,
    SbatchSettings,
    SrunSettings,
)


# Test creating a job step
@pytest.mark.parametrize(
    "settings_type, step_type",
    [
        pytest.param(
            AprunSettings,
            AprunStep,
            id=f"aprun",
        ),
        pytest.param(
            JsrunSettings,
            JsrunStep,
            id=f"jsrun",
        ),
        pytest.param(
            SrunSettings,
            SrunStep,
            id="srun",
        ),
        pytest.param(
            RunSettings,
            LocalStep,
            id="local",
        ),
    ],
)
def test_instantiate_run_settings(settings_type, step_type):
    run_settings = settings_type()
    run_settings.in_batch = True
    model = Model(
        exe="echo", exe_args="hello", name="model_name", run_settings=run_settings
    )
    jobStep = step_type(entity=model, run_settings=model.run_settings)
    assert jobStep.run_settings == run_settings
    assert jobStep.entity == model
    assert jobStep.entity_name == model.name
    assert jobStep.cwd == model.path
    assert jobStep.step_settings == model.run_settings


# Test creating a mpi job step
@pytest.mark.parametrize(
    "settings_type, step_type",
    [
        pytest.param(
            OrterunSettings,
            OrterunStep,
            id="orterun",
        ),
        pytest.param(
            MpirunSettings,
            MpirunStep,
            id="mpirun",
        ),
    ],
)
def test_instantiate_mpi_run_settings(settings_type, step_type):
    run_settings = settings_type(fail_if_missing_exec=False)
    run_settings.in_batch = True
    model = Model(
        exe="echo", exe_args="hello", name="model_name", run_settings=run_settings
    )
    jobStep = step_type(entity=model, run_settings=model.run_settings)
    assert jobStep.run_settings == run_settings
    assert jobStep.entity == model
    assert jobStep.entity_name == model.name
    assert jobStep.cwd == model.path
    assert jobStep.step_settings == model.run_settings


# Test creating a batch job step
@pytest.mark.parametrize(
    "settings_type, batch_settings_type, step_type",
    [
        pytest.param(
            JsrunSettings,
            BsubBatchSettings,
            BsubBatchStep,
            id=f"bsub",
        ),
        pytest.param(
            SrunSettings,
            SbatchSettings,
            SbatchStep,
            id="sbatch",
        ),
        pytest.param(
            RunSettings,
            QsubBatchSettings,
            QsubBatchStep,
            id="qsub",
        ),
    ],
)
def test_instantiate_batch_settings(settings_type, batch_settings_type, step_type):
    run_settings = settings_type()
    run_settings.in_batch = True
    batch_settings = batch_settings_type()
    model = Application(
        exe="echo",
        exe_args="hello",
        name="model_name",
        run_settings=run_settings,
        batch_settings=batch_settings,
    )
    jobStep = step_type(entity=model, batch_settings=model.batch_settings)
    assert jobStep.batch_settings == batch_settings
    assert jobStep.entity == model
    assert jobStep.entity_name == model.name
    assert jobStep.cwd == model.path
    assert jobStep.step_settings == model.batch_settings
