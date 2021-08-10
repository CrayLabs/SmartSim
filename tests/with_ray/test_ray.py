import logging
import sys
from os import environ

import psutil
import pytest

from smartsim import Experiment
from smartsim.error import SSUnsupportedError
from smartsim.ext.ray import RayCluster

"""Test Ray cluster local launch and shutdown.
"""

environ["OMP_NUM_THREADS"] = "1"
try:
    import ray
    import ray.util
except ImportError:
    pass

pytestmark = pytest.mark.skip(reason="Local launch is currently disabled for Ray")

# pytestmark = pytest.mark.skipif(
#     ("ray" not in sys.modules),
#     reason="requires Ray",
# )


def test_ray_local_launch_and_shutdown(fileutils, caplog):
    """Start a local (single node) Ray cluster and
    shut it down.
    """
    # Avoid Ray output
    caplog.set_level(logging.CRITICAL)

    test_dir = fileutils.make_test_dir("test-ray-local-launch-and-shutdown")

    exp = Experiment("ray-cluster", launcher="local", exp_path=test_dir)
    cluster = RayCluster(
        name="ray-cluster",
        run_args={},
        launcher="local",
        ray_port=6830,
        workers=0,
        batch=True,
        ray_args={"num-cpus": "4", "dashboard-port": "8266"},
        password=None,
    )
    exp.generate(cluster, overwrite=False)
    exp.start(cluster, block=False, summary=False)

    ray.init("ray://" + cluster.get_head_address() + ":10001")

    right_size = len(ray.nodes()) == 1
    if not right_size:
        ray.shutdown()
        exp.stop(cluster)
        assert False

    right_resources = ray.cluster_resources()["CPU"] == 4
    if not right_resources:
        ray.shutdown()
        exp.stop(cluster)
        assert False

    # Even setting batch to True must result in cluster.batch==False on local
    if cluster.batch:
        ray.shutdown()
        exp.stop(cluster)
        assert False

    ray.shutdown()
    exp.stop(cluster)

    raylet_active = False
    for proc in psutil.process_iter():
        try:
            if "raylet" in proc.name().lower():
                raylet_active = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    assert not raylet_active

    assert cluster.get_dashboard_address() == cluster.get_head_address() + ":8266"


def test_ray_errors(fileutils):
    """Try to start a local Ray cluster with incorrect settings."""

    test_dir = fileutils.make_test_dir("ray_test")

    with pytest.raises(SSUnsupportedError):
        _ = RayCluster(
            name="ray-cluster",
            path=test_dir,
            run_args={},
            launcher="notsupportedlauncher",
            workers=1,
        )
