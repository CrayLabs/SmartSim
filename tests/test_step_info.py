from smartsim import status
from smartsim._core.launcher.stepInfo import *


def test_str():
    step_info = StepInfo(
        status=status.STATUS_COMPLETED, launcher_status="COMPLETED", returncode=0
    )
    expected_output = "Status: Completed | Launcher Status COMPLETED | Returncode 0"

    assert str(step_info) == expected_output


def test_default():
    step_info = UnmanagedStepInfo()

    assert step_info._get_smartsim_status(None) == status.STATUS_FAILED
