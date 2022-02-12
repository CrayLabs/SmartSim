import pytest

from smartsim.database import SlurmOrchestrator, LSFOrchestrator, CobaltOrchestrator, PBSOrchestrator

tf_available = True
try:
    import tensorflow
except ImportError:
    tf_available = False

def test_deprecated_orchestrators(wlmutils):
    with pytest.deprecated_call():
        _ = SlurmOrchestrator(interface=wlmutils.get_test_interface())

    with pytest.deprecated_call():
        _ = LSFOrchestrator(interface=wlmutils.get_test_interface())

    with pytest.deprecated_call():
        _ = CobaltOrchestrator(interface=wlmutils.get_test_interface())

    with pytest.deprecated_call():
        _ = PBSOrchestrator(interface=wlmutils.get_test_interface())

@pytest.mark.skipif(not tf_available, reason="Requires TF to run")
def test_deprecated_tf():
    with pytest.deprecated_call():
        from smartsim.tf import freeze_model

def test_deprecated_constants():
    with pytest.deprecated_call():
        from smartsim import constants