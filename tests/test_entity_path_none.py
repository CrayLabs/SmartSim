# BSD 2-Clause License
#
# Copyright (c) 2021-2025, Hewlett Packard Enterprise
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

"""Regression tests for CrayLabs/SmartSim#556.

When path=None is passed to Model, Ensemble, or Orchestrator, the .path
attribute must be the current working directory — never the string 'None'.
"""

from os import getcwd

import pytest

from smartsim.entity import Ensemble, Model
from smartsim.settings import RunSettings
from smartsim.settings.base import BatchSettings


def test_model_path_none_defaults_to_cwd():
    """Model(path=None) must set .path to getcwd(), not 'None'."""
    rs = RunSettings("echo", ["hello"])
    m = Model("test-model", {}, rs, path=None)
    assert m.path != "None", "path must not be the string 'None'"
    assert m.path == getcwd()


def test_model_explicit_path_unchanged():
    """Passing an explicit path string must be stored as-is."""
    rs = RunSettings("echo", ["hello"])
    m = Model("test-model", {}, rs, path="/tmp/my-experiment")
    assert m.path == "/tmp/my-experiment"


def test_ensemble_path_none_defaults_to_cwd():
    """Ensemble(path=None) must set .path to getcwd(), not 'None'.

    A batch-only ensemble (no run_settings, no params) is used so the
    constructor does not attempt to spawn Model members, keeping the test
    free of HPC-specific launcher dependencies.
    """

    class _StubBatchSettings(BatchSettings):
        """Minimal BatchSettings subclass that requires no external commands."""

        def set_walltime(self, walltime: str) -> None:  # type: ignore[override]
            pass

        def set_nodes(self, num_nodes: int) -> None:
            pass

        def set_hostlist(self, host_list):
            pass

        def format_batch_args(self):
            return []

    bs = _StubBatchSettings()
    ens = Ensemble("test-ens", {}, path=None, batch_settings=bs)
    assert ens.path != "None", "path must not be the string 'None'"
    assert ens.path == getcwd()


def test_ensemble_explicit_path_unchanged():
    """Passing an explicit path string to Ensemble must be stored as-is."""

    class _StubBatchSettings(BatchSettings):
        def set_walltime(self, walltime: str) -> None:  # type: ignore[override]
            pass

        def set_nodes(self, num_nodes: int) -> None:
            pass

        def set_hostlist(self, host_list):
            pass

        def format_batch_args(self):
            return []

    bs = _StubBatchSettings()
    ens = Ensemble("test-ens", {}, path="/tmp/my-ensemble", batch_settings=bs)
    assert ens.path == "/tmp/my-ensemble"
