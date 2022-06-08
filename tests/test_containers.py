import pytest
from shutil import which

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings.containers import Singularity

# Check if singularity is available as command line tool
singularity_exists = which('singularity') is not None

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_basic(fileutils):
    # TODO: migrate from other test
    pass

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_args_str(fileutils):
    # TODO: migrate from other test
    pass

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_args_list(fileutils):
    # TODO: migrate from other test
    pass

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_mount_with_args(fileutils):
    # TODO: migrate from other test
    pass

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_mount_str(fileutils):
    # TODO: migrate from other test
    pass

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_mount_list(fileutils):
    # TODO: migrate from other test
    pass

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_mount_dict(fileutils):
    # TODO: migrate from other test
    pass


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_smartredis(fileutils, wlmutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.

    Note: This is a containerized port of test_smartredis.py
    """

    test_dir = fileutils.make_test_dir()
    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity('docker://benalbrecht10/smartsim-testing:latest')

    rs = exp.create_run_settings("python3", "producer.py --exchange", container=container)
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
        name="producer",
        params=params,
        run_settings=rs,
        perm_strat="step",
    )

    ensemble.register_incoming_entity(ensemble["producer_0"])
    ensemble.register_incoming_entity(ensemble["producer_1"])

    config = fileutils.get_test_conf_path("smartredis")
    ensemble.attach_generator_files(to_copy=[config])

    exp.generate(ensemble)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # client ensemble failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

