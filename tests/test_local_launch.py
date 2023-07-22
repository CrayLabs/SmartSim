# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
import typing as t
from functools import partial
from subprocess import PIPE

from smartsim import Experiment, status

"""
Test the launch of simple entity types with local launcher
"""


def test_models(fileutils):
    exp_name = "test-models-local-launch"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=1")

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings)

    exp.start(M1, M2, block=True, summary=True)
    statuses = exp.get_status(M1, M2)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def test_ensemble(fileutils):
    exp_name = "test-ensemble-launch"
    exp = Experiment(exp_name, launcher="local")
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")
    settings = exp.create_run_settings("python", f"{script} --time=1")

    ensemble = exp.create_ensemble("e1", run_settings=settings, replicas=2)
    ensemble.set_path(test_dir)

    exp.start(ensemble, block=True, summary=True)
    statuses = exp.get_status(ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


def mock_exec(
        cmd_list: t.List[str],
        cwd: str = "",
        env: t.Optional[t.Dict[str, t.Optional[str]]] = None,
        out: int = PIPE,
        err: int = PIPE,
        prefix: t.Iterable[str] = None,
        result: t.Iterable[str] = None,
) -> str:
    # use the prefix to differentiate repeated calls
    prefix = next(prefix, "")
    for key, value in env.items():
        print(f"_{prefix}_{key}={value}")
    return next(result, "")


def mock_detect(launcher: str) -> str:
    if launcher == "local":
        return ""
    return "mpiexec"


@pytest.mark.parametrize("launcher,alloc", [
        pytest.param("local", None, id="local"),
        pytest.param("lsf", "LSB_JOBID", id="lsf"),
        pytest.param("cobalt", "COBALT_JOBID", id="cobalt"),
        pytest.param("slurm", "SLURM_JOBID", id="slurm"),
        pytest.param("pbs", "PBS_JOBID", id="pbs"),
])
def test_env(fileutils, monkeypatch, capsys, launcher, alloc):
    """Ensure that the proper environment is passed to a launched step"""
    exp_name = "test-env-launch"

    monkeypatch.setattr("smartsim.settings.settings.detect_command", mock_detect)
    monkeypatch.setattr("smartsim.database.orchestrator.detect_command", mock_detect)
    monkeypatch.setattr("smartsim.settings.mpiSettings.MpiexecSettings._check_mpiexec_support", lambda *args, **kwargs: print('mock_check_mpiexec_support '))

    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir()

    script = fileutils.get_test_conf_path("sleep.py")

    # settings1 unique due to having env1 env vars
    env1 = {"keyonly1": "", "key1": "v1", "shared": "sharedvalue"}
    settings1 = exp.create_run_settings(exe="python",
                                        exe_args=f"{script} --time=0.1",
                                        env_vars=env1,
                                        fail_if_missing_exec=False)

    # settings2 unique due to having env2 env vars
    env2 = {"keyonly2": "", "key2": "v2", "shared": "sharedvalue", "ow": "override2"}
    settings2 = exp.create_run_settings(exe="python", 
                                        exe_args=f"{script} --time=0.1",
                                        env_vars=env2,
                                        fail_if_missing_exec=False)
    
    # settings3 unique due to having no env vars attached
    settings3 = exp.create_run_settings(exe="python", 
                                        exe_args=f"{script} --time=0.1",
                                        env_vars=None,
                                        fail_if_missing_exec=False) # demo that globals still work

    M1 = exp.create_model("m1", path=test_dir, run_settings=settings1)
    M2 = exp.create_model("m2", path=test_dir, run_settings=settings2)
    M3 = exp.create_model("m3", path=test_dir, run_settings=settings3)

    tasks = [M1, M2, M3]

    num_tasks = len(tasks)
    prefixes = iter(range(num_tasks))
    results = iter(["mockpid"] * num_tasks)
    partial_mock = partial(mock_exec, prefix=prefixes, result=results)
    monkeypatch.setattr("smartsim._core.launcher.taskManager.TaskManager.start_task", partial_mock)
    monkeypatch.setattr("smartsim._core.launcher.taskManager.TaskManager.start", lambda x: None)
    monkeypatch.setattr("smartsim._core.control.controller.SlurmLauncher.check_for_slurm", lambda x: None)
    
    os_environ = {"globalkey": "globalvalue", "ow": "globalow"}
    if alloc:
        os_environ[alloc] = "mock-alloc"

    monkeypatch.setattr("os.environ", os_environ)

    captured = capsys.readouterr()  # throw away existing output

    exp.start(*tasks, block=True, summary=True)
    captured = capsys.readouterr()

    # show unique keys applied to each step M1 / M2
    assert "_0_keyonly1=" in captured.out  # M1 has keyonly1
    assert "_1_keyonly1=" not in captured.out # M2 doesn't have keyonly1

    assert "_0_keyonly2=" not in captured.out # M1 doesn't have keyonly2
    assert "_1_keyonly2=" in captured.out # M2 has keyonly2
    
    assert "_0_key1=v1" in captured.out  # M1 has key1
    assert "_1_key1=v1" not in captured.out  # M2 doesn't have key1

    assert "_0_key2=v2" not in captured.out  # M1 doesn't have key2
    assert "_1_key2=v2" in captured.out  # M2 has key2

    # show same shared value for all steps
    assert "_0_shared=sharedvalue" in captured.out  # M1
    assert "_1_shared=sharedvalue" in captured.out  # M2

    # ensure shared value isn't in a task that had no env vars passed in
    assert "_2_shared=sharedvalue" not in captured.out  # M3
    
    # show the whole env was loaded in with the per-step env vars
    assert "_0_globalkey=globalvalue" in captured.out  # M1
    assert "_1_globalkey=globalvalue" in captured.out  # M2
    assert "_2_globalkey=globalvalue" in captured.out  # M3

    # show that M1/M3 didn't override a global env value
    assert "_0_ow=globalow" in captured.out
    assert "_2_ow=globalow" in captured.out

    # show that M2 DID override a global env value
    assert "_1_ow=override2" in captured.out
