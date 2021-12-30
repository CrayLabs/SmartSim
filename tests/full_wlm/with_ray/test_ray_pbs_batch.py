import logging
import time
from os import environ

import pytest

from smartsim import Experiment
from smartsim.exp.ray import RayCluster

"""Test Ray cluster launch and shutdown.
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
    if launcher != "pbs":
        pytest.skip("Test only runs on systems with PBS as WLM")

    caplog.set_level(logging.CRITICAL)
    test_dir = fileutils.make_test_dir("test-ray-pbs-launch-and-shutdown-batch")

    exp = Experiment("ray-cluster", test_dir, launcher=launcher)
    cluster = RayCluster(
        name="ray-cluster",
        run_args={},
        ray_args={"num-cpus": 4},
        launcher=launcher,
        num_nodes=2,
        alloc=None,
        batch=True,
        ray_port=6830,
        time="00:05:00",
        interface=wlmutils.get_test_interface(),
    )

    exp.generate(cluster)
    exp.start(cluster, block=False, summary=True)
    ctx = ray.client(cluster.get_head_address() + ":10001").connect()

    right_resources = False
    trials = 10
    while not right_resources and trials > 0:
        right_resources = (len(ray.nodes()), ray.cluster_resources()["CPU"]) == (2, 8)
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
