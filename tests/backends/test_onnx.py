import os
from pathlib import Path

import pytest

from smartsim import Experiment
from smartsim._core.utils import installed_redisai_backends
from smartsim.status import STATUS_FAILED

sklearn_available = True
try:
    from skl2onnx import to_onnx
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

except ImportError:
    sklearn_available = False


onnx_backend_available = "onnxruntime" in installed_redisai_backends()

should_run = sklearn_available and onnx_backend_available

pytestmark = pytest.mark.skipif(
    not should_run,
    reason="Requires scikit-learn, onnxmltools, skl2onnx and RedisAI onnx backend",
)


def test_sklearn_onnx(fileutils, mlutils, wlmutils):
    """This test needs two free nodes, 1 for the db and 1 some sklearn models

     here we test the following sklearn models:
       - LinearRegression
       - KMeans
       - RandomForestRegressor

    this test can run on CPU/GPU by setting SMARTSIM_TEST_DEVICE=GPU
    Similarly, the test can excute on any launcher by setting SMARTSIM_TEST_LAUNCHER
    which is local by default.

    Currently SmartSim only runs ONNX on GPU if libm.so with GLIBC_2.27 is present
    on the system. See: https://github.com/RedisAI/RedisAI/issues/826

    You may need to put CUDNN in your LD_LIBRARY_PATH if running on GPU
    """

    exp_name = "test_sklearn_onnx"
    test_dir = fileutils.make_test_dir(exp_name)
    exp = Experiment(exp_name, exp_path=test_dir, launcher=wlmutils.get_test_launcher())
    test_device = mlutils.get_test_device()

    db = wlmutils.get_orchestrator(nodes=1, port=6780)
    db.set_path(test_dir)
    exp.start(db)

    run_settings = exp.create_run_settings(
        "python", f"run_sklearn_onnx.py --device={test_device}"
    )
    model = exp.create_model("onnx_models", run_settings)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = Path(script_dir, "run_sklearn_onnx.py").resolve()
    model.attach_generator_files(to_copy=str(script_path))
    exp.generate(model)

    exp.start(model, block=True)

    exp.stop(db)
    # if model failed, test will fail
    model_status = exp.get_status(model)
    assert model_status[0] != STATUS_FAILED
