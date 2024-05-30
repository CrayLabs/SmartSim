# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import os
import pathlib
import time

import pytest

from smartsim import Experiment

if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_batch_application_and_ensemble(test_dir, wlmutils):
    exp_name = "test-batch"
    launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)
    rs = exp.create_run_settings("echo", ["spam", "eggs"])
    bs = exp.create_batch_settings()

    test_application = exp.create_application(
        "test_application", path=test_dir, run_settings=rs, batch_settings=bs
    )
    exp.generate(test_application)
    exp.start(test_application, block=True)

    assert pathlib.Path(test_application.path).exists()
    _should_be_symlinked(
        pathlib.Path(test_application.path, f"{test_application.name}.out"), True
    )
    _should_be_symlinked(
        pathlib.Path(test_application.path, f"{test_application.name}.err"), False
    )
    _should_not_be_symlinked(
        pathlib.Path(test_application.path, f"{test_application.name}.sh")
    )

    test_ensemble = exp.create_ensemble(
        "test_ensemble", params={}, batch_settings=bs, run_settings=rs, replicas=3
    )
    exp.generate(test_ensemble)
    exp.start(test_ensemble, block=True)

    assert pathlib.Path(test_ensemble.path).exists()
    for i in range(len(test_ensemble.applications)):
        _should_be_symlinked(
            pathlib.Path(
                test_ensemble.path,
                f"{test_ensemble.name}_{i}",
                f"{test_ensemble.name}_{i}.out",
            ),
            True,
        )
        _should_be_symlinked(
            pathlib.Path(
                test_ensemble.path,
                f"{test_ensemble.name}_{i}",
                f"{test_ensemble.name}_{i}.err",
            ),
            False,
        )

    _should_not_be_symlinked(pathlib.Path(exp.exp_path, "smartsim_params.txt"))


def test_batch_ensemble_symlinks(test_dir, wlmutils):
    exp_name = "test-batch-ensemble"
    launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)
    rs = exp.create_run_settings("echo", ["spam", "eggs"])
    bs = exp.create_batch_settings()
    test_ensemble = exp.create_ensemble(
        "test_ensemble", params={}, batch_settings=bs, run_settings=rs, replicas=3
    )
    exp.generate(test_ensemble)
    exp.start(test_ensemble, block=True)

    for i in range(len(test_ensemble.applications)):
        _should_be_symlinked(
            pathlib.Path(
                test_ensemble.path,
                f"{test_ensemble.name}_{i}",
                f"{test_ensemble.name}_{i}.out",
            ),
            True,
        )
        _should_be_symlinked(
            pathlib.Path(
                test_ensemble.path,
                f"{test_ensemble.name}_{i}",
                f"{test_ensemble.name}_{i}.err",
            ),
            False,
        )

    _should_not_be_symlinked(pathlib.Path(exp.exp_path, "smartsim_params.txt"))


def test_batch_application_symlinks(test_dir, wlmutils):
    exp_name = "test-batch-application"
    launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)
    rs = exp.create_run_settings("echo", ["spam", "eggs"])
    bs = exp.create_batch_settings()
    test_application = exp.create_application(
        "test_application", path=test_dir, run_settings=rs, batch_settings=bs
    )
    exp.generate(test_application)
    exp.start(test_application, block=True)

    assert pathlib.Path(test_application.path).exists()

    _should_be_symlinked(
        pathlib.Path(test_application.path, f"{test_application.name}.out"), True
    )
    _should_be_symlinked(
        pathlib.Path(test_application.path, f"{test_application.name}.err"), False
    )
    _should_not_be_symlinked(
        pathlib.Path(test_application.path, f"{test_application.name}.sh")
    )


def test_batch_feature_store_symlinks(test_dir, wlmutils):
    exp_name = "test-batch-orc"
    launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)
    port = 2424
    db = exp.create_feature_store(
        fs_nodes=3,
        port=port,
        batch=True,
        interface=wlmutils.get_test_interface(),
        single_cmd=False,
    )
    exp.generate(db)
    exp.start(db, block=True)
    time.sleep(2)
    exp.stop(db)

    _should_be_symlinked(pathlib.Path(db.path, f"{db.name}.out"), False)
    _should_be_symlinked(pathlib.Path(db.path, f"{db.name}.err"), False)

    for i in range(db.fs_nodes):
        _should_be_symlinked(pathlib.Path(db.path, f"{db.name}_{i}.out"), False)
        _should_be_symlinked(pathlib.Path(db.path, f"{db.name}_{i}.err"), False)
        _should_not_be_symlinked(
            pathlib.Path(db.path, f"nodes-orchestrator_{i}-{port}.conf")
        )


def _should_not_be_symlinked(non_linked_path: pathlib.Path):
    """Helper function for assertions about paths that should NOT be symlinked"""
    assert non_linked_path.exists()
    assert not non_linked_path.is_symlink()


def _should_be_symlinked(linked_path: pathlib.Path, open_file: bool):
    """Helper function for assertions about paths that SHOULD be symlinked"""
    assert linked_path.exists()
    assert linked_path.is_symlink()
    # ensure the source file exists
    assert pathlib.Path(os.readlink(linked_path)).exists()
    if open_file:
        with open(pathlib.Path(os.readlink(linked_path)), "r") as file:
            log_contents = file.read()
        assert "spam eggs" in log_contents
