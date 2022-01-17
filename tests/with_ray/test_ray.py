import logging
import time
from os import environ

import psutil
import pytest

from smartsim import Experiment, slurm
from smartsim.error import SSUnsupportedError
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


def test_ray_launch_and_shutdown_wlm(fileutils, wlmutils, caplog):
    launcher = wlmutils.get_test_launcher()
    if launcher == "local":
        pytest.skip("Test can not be run on local launcher")

    caplog.set_level(logging.CRITICAL)
    test_dir = fileutils.make_test_dir("test-ray-launch-and-shutdown-wlm")

    exp = Experiment("ray-cluster", test_dir, launcher=launcher)
    cluster = RayCluster(
        name="ray-cluster",
        run_args={},
        ray_args={"num-cpus": 4},
        launcher=launcher,
        num_nodes=2,
        alloc=None,
        batch=False,
        time="00:05:00",
        interface=wlmutils.get_test_interface(),
    )

    exp.generate(cluster)
    exp.start(cluster, block=False, summary=False)
    ctx = ray.init("ray://" + cluster.get_head_address() + ":10001")

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


def test_ray_launch_and_shutdown_in_alloc(fileutils, wlmutils, caplog):
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip("Test only runs on systems with Slurm as WLM")
    if "SLURM_JOBID" in environ:
        pytest.skip("Test can not be run inside an allocation")

    caplog.set_level(logging.CRITICAL)
    test_dir = fileutils.make_test_dir("test-ray-slurm-launch-and-shutdown-in-alloc")

    alloc = slurm.get_allocation(4, time="00:05:00")

    exp = Experiment("ray-cluster", test_dir, launcher=launcher)
    cluster = RayCluster(
        name="ray-cluster",
        run_args={},
        ray_args={"num-cpus": 4, "dashboard-port": "8266"},
        launcher=launcher,
        workers=2,
        alloc=alloc,
        batch=False,
        interface=wlmutils.get_test_interface(),
    )

    exp.generate(cluster)
    exp.start(cluster, block=False, summary=False)
    ctx = ray.init("ray://" + cluster.get_head_address() + ":10001")

    right_resources = False
    trials = 10
    while not right_resources and trials > 0:
        right_resources = (len(ray.nodes()), ray.cluster_resources()["CPU"]) == (3, 12)
        trials -= 1
        time.sleep(1)

    assert cluster.get_dashboard_address() == cluster.get_head_address() + ":8266"

    if not right_resources:
        ctx.disconnect()
        ray.shutdown()
        exp.stop(cluster)
        slurm.release_allocation(alloc)
        assert False

    ctx.disconnect()
    ray.shutdown()
    exp.stop(cluster)
    slurm.release_allocation(alloc)


def test_ray_errors(fileutils):
    """Try to start a local Ray cluster with incorrect settings."""

    test_dir = fileutils.make_test_dir("test-ray-errors")

    with pytest.raises(SSUnsupportedError):
        _ = RayCluster(
            name="local-ray-cluster",
            path=test_dir,
            run_args={},
            launcher="local",
            num_nodes=1,
        )

    with pytest.raises(ValueError):
        _ = RayCluster(
            name="small-ray-cluster",
            path=test_dir,
            run_args={},
            launcher="slurm",
            num_nodes=0,
        )


@pytest.mark.skip(reason="Local launch is currently disabled for Ray")
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
        num_nodes=1,
        batch=True,
        ray_args={"num-cpus": "4", "dashboard-port": "8266"},
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
