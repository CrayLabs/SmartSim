import os
from pathlib import Path

import pytest

from smartsim import Experiment
from smartsim.constants import STATUS_FAILED
from smartsim.error import SmartSimError

should_run = True
try:
    from tensorflow import keras

    from smartsim.tf import freeze_model
except (ImportError, SmartSimError):
    should_run = False

pytestmark = pytest.mark.skipif(
    not should_run,
    reason="requires tensorflow/keras",
)


def test_keras_model(fileutils, mlutils, wlmutils):
    """This test needs two free nodes, 1 for the db and 1 for a keras model script

    this test can run on CPU/GPU by setting SMARTSIM_TEST_DEVICE=GPU
    Similarly, the test can excute on any launcher by setting SMARTSIM_TEST_LAUNCHER
    which is local by default.

    You may need to put CUDNN in your LD_LIBRARY_PATH if running on GPU
    """

    exp_name = "test_keras_model"
    test_dir = fileutils.make_test_dir(exp_name)
    exp = Experiment(exp_name, exp_path=test_dir, launcher=wlmutils.get_test_launcher())
    test_device = mlutils.get_test_device()

    db = wlmutils.get_orchestrator(nodes=1, port=6780)
    db.set_path(test_dir)
    exp.start(db)

    run_settings = exp.create_run_settings(
        "python", f"run_tf.py --device={test_device}"
    )
    model = exp.create_model("tf_script", run_settings)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = Path(script_dir, "run_tf.py").resolve()
    model.attach_generator_files(to_copy=str(script_path))
    exp.generate(model)

    exp.start(model, block=True)

    exp.stop(db)
    # if model failed, test will fail
    model_status = exp.get_status(model)
    assert model_status != STATUS_FAILED
