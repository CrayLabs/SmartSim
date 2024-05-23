from smartsim.entity import Ensemble
from smartsim.settings import RunSettings


def test_create_ensemble():
    run_settings = RunSettings()
    ensemble = Ensemble(
        name="model",
        exe="echo",
        run_settings=run_settings,
        exe_args=["hello"],
        replicas=2,
    )
    assert ensemble.exe == "echo"
    assert ensemble.exe_args == ["hello"]
    for model in ensemble:
        assert model.exe == ["/usr/bin/echo"]
        assert model.exe_args == ["hello"]
