import pytest

from smartsim.database import SlurmOrchestrator, LSFOrchestrator, CobaltOrchestrator, PBSOrchestrator

tf_available = True
try:
    import tensorflow
except ImportError:
    tf_available = False

def test_deprecated_orchestrators():
    with pytest.deprecated_call():
        _ = SlurmOrchestrator()

    with pytest.deprecated_call():
        _ = LSFOrchestrator()

    with pytest.deprecated_call():
        _ = CobaltOrchestrator()

    with pytest.deprecated_call():
        _ = PBSOrchestrator()

@pytest.mark.skipif(not tf_available, reason="Requires TF to run")
def test_deprecated_tf():
    with pytest.deprecated_call():
        from smartsim.tf import freeze_model
