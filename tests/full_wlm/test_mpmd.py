import pytest

from smartsim._core.utils.helpers import is_valid_cmd
from copy import deepcopy
from pprint import pprint
from smartsim import Experiment, status


# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_slurm_mpmd(fileutils, wlmutils):
    """Run an MPMD model twice 
    
    and check that it always gets executed the same way.
    All MPMD-compatible run commands which do not
    require MPI are tested.

    This test requires three nodes to run.
    """
    exp_name = "test-slurm-mpmd"
    launcher = wlmutils.get_test_launcher()
    # MPMD is supported in LSF, but the test for it is different
    mpmd_supported = ["slurm", "pbs", "cobalt"]
    if launcher not in mpmd_supported:
        pytest.skip("Test requires Slurm, PBS, or Cobalt to run")

    # aprun returns an error if the launched app is not an MPI exec
    # as we do not want to add mpi4py as a dependency, we prefer to
    # skip this test for aprun
    by_launcher = {
        "slurm": ["srun", "mpirun"],
        "pbs": ["mpirun"],
        "cobalt": ["mpirun"],
    }

    exp = Experiment(exp_name, launcher=launcher)

    def prune_commands(launcher):
        available_commands = []
        if launcher in by_launcher:
            for cmd in by_launcher[launcher]:
                if is_valid_cmd(cmd):
                    available_commands.append(cmd)
        return available_commands

    run_commands = prune_commands(launcher)
    if len(run_commands) == 0:
        pytest.skip(f"MPMD on {launcher} only supported for run commands {by_launcher[launcher]}")

    test_dir = fileutils.make_test_dir(exp_name)
    for run_command in run_commands:
        script = fileutils.get_test_conf_path("sleep.py")
        settings = exp.create_run_settings("python", f"{script} --time=5", run_command=run_command)
        settings.set_tasks(1)

        settings.make_mpmd(deepcopy(settings))
        settings.make_mpmd(deepcopy(settings))

        mpmd_model = exp.create_model("mmpd", path=test_dir, run_settings=settings)
        exp.generate(mpmd_model, overwrite=True)
        exp.start(mpmd_model, block=True)
        statuses = exp.get_status(mpmd_model)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])

        exp.start(mpmd_model, block=True)
        statuses = exp.get_status(mpmd_model)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])

