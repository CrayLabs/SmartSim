from shutil import which
import pytest
import os
import json
from pathlib import Path
from smartsim.error.errors import LauncherError, SSUnsupportedError, SmartSimError

import smartsim.wlm as wlm


def test_get_hosts(alloc_specs):
    def verify_output(output):
        assert isinstance(output, list)
        if "host_list" in alloc_specs:
            assert output == alloc_specs["host_list"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_hosts())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_hosts()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ:
            verify_output(wlm.get_hosts())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_hosts()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_hosts()


def test_get_queue(alloc_specs):
    def verify_output(output):
        assert isinstance(output, str)
        if "queue" in alloc_specs:
            assert output == alloc_specs["queue"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_queue())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_queue()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ:
            verify_output(wlm.get_queue())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_queue()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_queue()


def test_get_tasks(alloc_specs):
    def verify_output(output):
        assert isinstance(output, int)
        if "num_tasks" in alloc_specs:
            assert output == alloc_specs["num_tasks"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_tasks())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ and which("qstat"):
            verify_output(wlm.get_tasks())
        elif "PBS_JOBID" in os.environ:
            with pytest.raises(LauncherError):
                wlm.get_tasks()
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_tasks()


def test_get_tasks_per_node(alloc_specs):
    def verify_output(output):
        assert isinstance(output, str)
        if "num_tasks" in alloc_specs:
            assert output == alloc_specs["num_tasks"]

    if pytest.test_launcher == "slurm":
        if "SLURM_JOBID" in os.environ:
            verify_output(wlm.get_tasks_per_node())
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks_per_node()

    elif pytest.test_launcher == "pbs":
        if "PBS_JOBID" in os.environ and which("qstat"):
            verify_output(wlm.get_tasks_per_node())
        elif "PBS_JOBID" in os.environ:
            with pytest.raises(LauncherError):
                wlm.get_tasks_per_node()
        else:
            with pytest.raises(SmartSimError):
                wlm.get_tasks_per_node()

    else:
        with pytest.raises(SSUnsupportedError):
            wlm.get_tasks_per_node()
