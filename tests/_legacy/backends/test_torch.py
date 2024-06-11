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
from pathlib import Path

import pytest

from smartsim import Experiment
from smartsim._core.utils import installed_redisai_backends
from smartsim.status import SmartSimStatus

torch_available = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch_available = False

torch_backend_available = "torch" in installed_redisai_backends()

should_run = torch_available and torch_backend_available
pytestmark = pytest.mark.skipif(
    not should_run, reason="Requires torch RedisAI torch backend"
)


def test_torch_model_and_script(
    wlm_experiment, prepare_fs, single_fs, mlutils, wlmutils
):
    """This test needs two free nodes, 1 for the fs and 1 for a torch model script

     Here we test both the torchscipt API and the NN API from torch

    this test can run on CPU/GPU by setting SMARTSIM_TEST_DEVICE=GPU
    Similarly, the test can excute on any launcher by setting SMARTSIM_TEST_LAUNCHER
    which is local by default.

    You may need to put CUDNN in your LD_LIBRARY_PATH if running on GPU
    """

    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_feature_store(fs.checkpoint_file)
    test_device = mlutils.get_test_device()

    run_settings = wlm_experiment.create_run_settings(
        "python", f"run_torch.py --device={test_device}"
    )
    if wlmutils.get_test_launcher() != "local":
        run_settings.set_tasks(1)
    model = wlm_experiment.create_application("torch_script", run_settings)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = Path(script_dir, "run_torch.py").resolve()
    model.attach_generator_files(to_copy=str(script_path))
    wlm_experiment.generate(model)

    wlm_experiment.start(model, block=True)

    # if model failed, test will fail
    model_status = wlm_experiment.get_status(model)[0]
    assert model_status != SmartSimStatus.STATUS_FAILED
