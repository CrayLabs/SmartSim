import logging
import os.path as osp
import sys
import time
from os import environ

import pytest

from smartsim import Experiment
from smartsim._core.launcher import slurm
from smartsim.exp.ray import RayCluster

"""Test Ray cluster Slurm launch and shutdown.
"""

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")

environ["OMP_NUM_THREADS"] = "1"
shouldrun = True
try:
    import ray
except ImportError:
    shouldrun = False


pytestmark = pytest.mark.skipif(
    not shouldrun,
    reason="requires Ray",
)


def test_ray_launch_and_shutdown_batch(fileutils, wlmutils, caplog):
    launcher = wlmutils.get_test_launcher()
    if launcher == "local":
        pytest.skip("Test cannot be run with local launcher")

    caplog.set_level(logging.CRITICAL)
    test_dir = fileutils.make_test_dir("test-ray-launch-and-shutdown-batch")

    exp = Experiment("ray-cluster", test_dir, launcher=launcher)
    cluster = RayCluster(
        name="ray-cluster",
        run_args={},
        ray_args={"num-cpus": 4},
        launcher=launcher,
        num_nodes=2,
        batch=True,
        interface=wlmutils.get_test_interface(),
        batch_args={"A": wlmutils.get_test_account(), "queue": "debug-flat-quad"}
        if launcher == "cobalt"
        else None,
        time="00:05:00",
    )

    exp.generate(cluster)

    try:
        exp.start(cluster, block=False, summary=True)
        ctx = ray.init("ray://" + cluster.get_head_address() + ":10001")

        right_resources = False
        trials = 10
        while not right_resources and trials > 0:
            right_resources = (len(ray.nodes()), ray.cluster_resources()["CPU"]) == (
                2,
                8,
            )
            trials -= 1
            time.sleep(1)

        if not right_resources:
            ctx.disconnect()
            ray.shutdown()
            exp.stop(cluster)
            assert False

        ctx.disconnect()
        ray.shutdown()
        exp.stop(cluster)
    except:
        # Catch all errors, most of which can come from Ray
        exp.stop(cluster)
        assert False
