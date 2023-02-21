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

import os.path as osp
import time

import pytest

from smartsim import status
from smartsim.database import Orchestrator
from smartsim.error.errors import SmartSimError
from smartsim.experiment import Experiment

shouldrun_tf = True
if shouldrun_tf:
    try:
        from tensorflow import keras

        from smartsim.ml.tf import DynamicDataGenerator as TFDataGenerator
    except:
        shouldrun_tf = False

shouldrun_torch = True
if shouldrun_torch:
    try:
        import torch

        from smartsim.ml.torch import DynamicDataGenerator as TorchDataGenerator
    except:
        shouldrun_torch = False


def create_uploader(experiment: Experiment, filedir, format):
    """Start an ensemble of two processes producing sample batches at
    regular intervals.
    """
    run_settings = experiment.create_run_settings(
        exe="python",
        exe_args=["data_uploader.py", f"--format={format}"],
        env_vars={"PYTHONUNBUFFERED": "1"},
    )

    uploader = experiment.create_ensemble(
        "test_uploader", replicas=2, run_settings=run_settings
    )

    uploader.attach_generator_files(to_copy=[osp.join(filedir, "data_uploader.py")])
    uploader.enable_key_prefixing()
    experiment.generate(uploader, overwrite=True)
    return uploader


def create_trainer_tf(experiment: Experiment, filedir):
    run_settings = experiment.create_run_settings(
        exe="python",
        exe_args=["training_service_tf.py"],
        env_vars={"PYTHONUNBUFFERED": "1"},
    )

    trainer = experiment.create_model("trainer", run_settings=run_settings)

    trainer.attach_generator_files(
        to_copy=[osp.join(filedir, "training_service_tf.py")]
    )
    experiment.generate(trainer, overwrite=True)
    return trainer


def create_trainer_torch(experiment: Experiment, filedir):
    run_settings = experiment.create_run_settings(
        exe="python",
        exe_args=["training_service_torch.py"],
        env_vars={"PYTHONUNBUFFERED": "1"},
    )

    trainer = experiment.create_model("trainer", run_settings=run_settings)

    trainer.attach_generator_files(
        to_copy=[osp.join(filedir, "training_service_torch.py")]
    )
    experiment.generate(trainer, overwrite=True)
    return trainer


def test_batch_dataloader_tf(fileutils, wlmutils):
    if not shouldrun_tf:
        pytest.skip("Test needs TensorFlow to run.")

    test_dir = fileutils.make_test_dir()
    exp = Experiment("test-batch-dataloader-tf", exp_path=test_dir)
    config_path = fileutils.get_test_conf_path("ml")
    dataloader = create_uploader(exp, config_path, "tf")
    trainer_tf = create_trainer_tf(exp, config_path)

    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc)
    exp.start(dataloader, block=False)

    for entity in dataloader:
        trainer_tf.register_incoming_entity(entity)

    exp.start(trainer_tf, block=True)
    if exp.get_status(trainer_tf)[0] != status.STATUS_COMPLETED:
        exp.stop(orc)
        assert False

    exp.stop(orc)

    trials = 5
    if exp.get_status(dataloader)[0] != status.STATUS_COMPLETED:
        time.sleep(5)
        trials -= 1
        if trials == 0:
            assert False


def test_batch_dataloader_torch(fileutils, wlmutils):
    if not shouldrun_torch:
        pytest.skip("Test needs PyTorch to run.")

    test_dir = fileutils.make_test_dir()
    exp = Experiment("test-batch-dataloader-tf", exp_path=test_dir)
    config_path = fileutils.get_test_conf_path("ml")
    dataloader = create_uploader(exp, config_path, "torch")
    trainer_torch = create_trainer_torch(exp, config_path)

    orc = Orchestrator(wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc)
    exp.start(dataloader, block=False)

    for entity in dataloader:
        trainer_torch.register_incoming_entity(entity)

    exp.start(trainer_torch, block=True)
    if exp.get_status(trainer_torch)[0] != status.STATUS_COMPLETED:
        exp.stop(orc)
        assert False

    exp.stop(orc)

    trials = 5
    if exp.get_status(dataloader)[0] != status.STATUS_COMPLETED:
        time.sleep(5)
        trials -= 1
        if trials == 0:
            assert False


@pytest.mark.skipif(
    not (shouldrun_torch or shouldrun_tf), reason="Requires TF or PyTorch"
)
def test_wrong_dataloaders(fileutils, wlmutils):
    test_dir = fileutils.make_test_dir()
    exp = Experiment("test-wrong-dataloaders", exp_path=test_dir)
    orc = Orchestrator(wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc)

    if shouldrun_tf:
        with pytest.raises(SmartSimError):
            _ = TFDataGenerator(address=orc.get_address()[0], cluster=False)

    if shouldrun_torch:
        with pytest.raises(SmartSimError):
            torch_data_gen = TorchDataGenerator(
                address=orc.get_address()[0], cluster=False
            )
            torch_data_gen.init_samples()

    exp.stop(orc)
