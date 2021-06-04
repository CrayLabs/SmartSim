import os.path as osp
from os import environ
import pickle
import sys
import logging
from shutil import rmtree
import psutil

import pytest

from smartsim import Experiment
from smartsim.error import SSUnsupportedError
from smartsim.ray import RayCluster

"""Test Ray cluster launch and shutdown.
"""

environ["OMP_NUM_THREADS"] = '1'
try:
    import ray
    import ray.util
except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    ("ray" not in sys.modules),
    reason="requires Ray",
    )

def test_ray_launch_and_shutdown(fileutils, caplog):
    """Start a local (single node) Ray cluster and
    shut it down.
    """
    # Avoid Ray output
    caplog.set_level(logging.CRITICAL)
    
    test_dir = fileutils.make_test_dir("ray_test")
    
    exp = Experiment("ray-cluster", launcher='local', exp_path=test_dir)
    cluster = RayCluster(name="ray-cluster", run_args={}, path=' ',
                         launcher='local', workers=0, batch=True, ray_num_cpus=4)
    exp.generate(cluster, overwrite=False)
    exp.start(cluster, block=False, summary=False)
    
    ray.util.connect(cluster.head_model.address +":10001")

    assert(len(ray.nodes())==1)
    assert(ray.cluster_resources()['CPU']==4)
    
    # Even setting batch to True must result in cluster.batch==False on local
    assert(not cluster.batch)
    
    exp.stop(cluster)
    
    raylet_active = False
    for proc in psutil.process_iter():
        try:
            if "raylet" in proc.name().lower():
                raylet_active = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    assert(not raylet_active)
    
def test_ray_errors(fileutils):
    """Try to start a local Ray cluster with incorrect settings."""
    
    test_dir = fileutils.make_test_dir("ray_test")
    exp = Experiment("ray-cluster", launcher='local', exp_path=test_dir)

    with pytest.raises(SSUnsupportedError):
        cluster = RayCluster(name="ray-cluster", run_args={}, path=' ',
                             launcher='local', workers=1)