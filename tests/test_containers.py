import pytest
from shutil import which

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings.containers import Singularity

REDIS_PORT = 6780

# Check if singularity is available as command line tool
singularity_exists = which('singularity') is not None

@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_redis(fileutils):
    """Run two processes in singularity containers, each process puts a tensor
    on the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.
    """

    test_dir = fileutils.make_test_dir("test_singularity_redis")
    exp = Experiment(
        "test_singularity_redis", exp_path=test_dir, launcher="auto"
    )

    # create and start a database
    orc = Orchestrator(port=REDIS_PORT)
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity('benalbrecht10/smartsim-testing')
    rs = exp.create_run_settings("python", "send_data.py", container=container)
    model = exp.create_model('send_data', rs)
    model.attach_generator_files(to_copy='test_configs/send_data.py')
    exp.generate(model, overwrite=True)
    exp.start(model, block=True, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(model)
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # model experiment failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

