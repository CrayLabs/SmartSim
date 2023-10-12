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

import os
from os import path as osp

import numpy as np
import pytest

from smartsim.database import Orchestrator
from smartsim.error.errors import SSInternalError
from smartsim.experiment import Experiment
from smartsim.ml.data import DataInfo, TrainingDataUploader
from smartsim.status import STATUS_COMPLETED

shouldrun_tf = True
if shouldrun_tf:
    try:
        from tensorflow import keras

        from smartsim.ml.tf import DynamicDataGenerator as TFDataGenerator
        from smartsim.ml.tf import StaticDataGenerator as TFStaticDataGenerator
    except:
        shouldrun_tf = False

shouldrun_torch = True
if shouldrun_torch:
    try:
        import torch

        from smartsim.ml.torch import DataLoader
        from smartsim.ml.torch import \
            DynamicDataGenerator as TorchDataGenerator
        from smartsim.ml.torch import \
            StaticDataGenerator as TorchStaticDataGenerator
    except:
        shouldrun_torch = False


def check_dataloader(dl, rank, dynamic):
    assert dl.list_name == "test_data_list"
    assert dl.sample_name == "test_samples"
    assert dl.target_name == "test_targets"
    assert dl.num_classes == 2
    assert dl.verbose == True
    assert dl.replica_rank == rank
    assert dl.num_replicas == 2
    assert dl.address == None
    assert dl.cluster == False
    assert dl.shuffle == True
    assert dl.batch_size == 4
    assert dl.autoencoding == False
    assert dl.need_targets == True
    assert dl.dynamic == dynamic


def run_local_uploaders(mpi_size, format="torch"):
    batch_size = 4
    for rank in range(mpi_size):
        os.environ["SSKEYOUT"] = f"test_uploader_{rank}"

        data_uploader = TrainingDataUploader(
            list_name="test_data_list",
            sample_name="test_samples",
            target_name="test_targets",
            num_classes=mpi_size,
            cluster=False,
            address=None,
            rank=rank,
            verbose=True,
        )

        assert data_uploader._info.list_name == "test_data_list"
        assert data_uploader._info.sample_name == "test_samples"
        assert data_uploader._info.target_name == "test_targets"
        assert data_uploader._info.num_classes == mpi_size
        assert data_uploader._info._ds_name == "test_data_list_info"

        if rank == 0:
            data_uploader.publish_info()

        batches_per_loop = 1
        shape = (
            (batch_size * batches_per_loop, 32, 32, 1)
            if format == "tf"
            else (batch_size * batches_per_loop, 1, 32, 32)
        )

        for _ in range(2):
            for mpi_rank in range(mpi_size):
                new_batch = np.random.normal(
                    loc=float(mpi_rank), scale=5.0, size=shape
                ).astype(float)
                new_labels = (
                    np.ones(shape=(batch_size * batches_per_loop,)).astype(int)
                    * mpi_rank
                )

                data_uploader.put_batch(new_batch, new_labels)

    return data_uploader._info


def train_tf(generator):
    if not shouldrun_tf:
        return

    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                filters=4,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(32, 32, 1),
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(generator.num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    for epoch in range(2):
        model.fit(
            generator,
            steps_per_epoch=None,
            epochs=epoch + 1,
            initial_epoch=epoch,
            batch_size=generator.batch_size,
            verbose=2,
        )


@pytest.mark.skipif(not shouldrun_tf, reason="Test needs TensorFlow to run")
def test_tf_dataloaders(fileutils, wlmutils):
    test_dir = fileutils.make_test_dir()
    exp = Experiment("test_tf_dataloaders", test_dir, launcher=wlmutils.get_test_launcher())
    orc: Orchestrator = wlmutils.get_orchestrator()
    exp.generate(orc)
    exp.start(orc)

    try:
        os.environ["SSDB"] = orc.get_address()[0]
        data_info = run_local_uploaders(mpi_size=2, format="tf")

        os.environ["SSKEYIN"] = "test_uploader_0,test_uploader_1"
        for rank in range(2):
            tf_dynamic = TFDataGenerator(
                data_info_or_list_name="test_data_list",
                cluster=False,
                verbose=True,
                num_replicas=2,
                replica_rank=rank,
                batch_size=4,
                max_fetch_trials=5,
                dynamic=False,  # catch wrong arg
            )
            train_tf(tf_dynamic)
            assert len(tf_dynamic) == 4
            check_dataloader(tf_dynamic, rank, dynamic=True)
        for rank in range(2):
            tf_static = TFStaticDataGenerator(
                data_info_or_list_name=data_info,
                cluster=False,
                verbose=True,
                num_replicas=2,
                replica_rank=rank,
                batch_size=4,
                max_fetch_trials=5,
                dynamic=True,  # catch wrong arg
            )
            train_tf(tf_static)
            assert len(tf_static) == 4
            check_dataloader(tf_static, rank, dynamic=False)

    except Exception as e:
        raise e
    finally:
        exp.stop(orc)
        os.environ.pop("SSDB", "")
        os.environ.pop("SSKEYIN", "")
        os.environ.pop("SSKEYOUT", "")


def create_trainer_torch(experiment: Experiment, filedir, wlmutils):
    run_settings = wlmutils.get_run_settings(
        exe="python",
        args=["training_service_torch.py"],
    )

    trainer = experiment.create_model("trainer", run_settings=run_settings)

    trainer.attach_generator_files(
        to_copy=[osp.join(filedir, "training_service_torch.py")]
    )
    experiment.generate(trainer, overwrite=True)
    return trainer


@pytest.mark.skipif(not shouldrun_torch, reason="Test needs Torch to run")
def test_torch_dataloaders(fileutils, wlmutils):
    test_dir = fileutils.make_test_dir()
    exp = Experiment("test_tf_dataloaders", test_dir, launcher=wlmutils.get_test_launcher())
    orc: Orchestrator = wlmutils.get_orchestrator()
    config_dir = fileutils.get_test_dir_path("ml")
    exp.generate(orc)
    exp.start(orc)

    try:
        os.environ["SSDB"] = orc.get_address()[0]
        data_info = run_local_uploaders(mpi_size=2)

        os.environ["SSKEYIN"] = "test_uploader_0,test_uploader_1"
        for rank in range(2):
            torch_dynamic = TorchDataGenerator(
                data_info_or_list_name="test_data_list",
                cluster=False,
                verbose=True,
                num_replicas=2,
                replica_rank=rank,
                batch_size=4,
                max_fetch_trials=5,
                dynamic=False,  # catch wrong arg
                init_samples=True,  # catch wrong arg
            )
            check_dataloader(torch_dynamic, rank, dynamic=True)

            torch_dynamic.init_samples(5)
            for _ in range(2):
                for _ in torch_dynamic:
                    continue

        for rank in range(2):
            torch_static = TorchStaticDataGenerator(
                data_info_or_list_name=data_info,
                cluster=False,
                verbose=True,
                num_replicas=2,
                replica_rank=rank,
                batch_size=4,
                max_fetch_trials=5,
                dynamic=True,  # catch wrong arg
                init_samples=True,  # catch wrong arg
            )
            check_dataloader(torch_static, rank, dynamic=False)

            torch_static.init_samples(5)
            for _ in range(2):
                for _ in torch_static:
                    continue
        
        trainer = create_trainer_torch(exp, config_dir, wlmutils)
        exp.start(trainer, block=True)
        
        assert exp.get_status(trainer)[0] == STATUS_COMPLETED

    except Exception as e:
        raise e
    finally:
        exp.stop(orc)
        os.environ.pop("SSDB", "")
        os.environ.pop("SSKEYIN", "")
        os.environ.pop("SSKEYOUT", "")


def test_data_info_repr():
    data_info = DataInfo(
        list_name="a_list", sample_name="the_samples", target_name=None
    )
    data_info_repr = "DataInfo object\n"
    data_info_repr += "Aggregation list name: a_list\n"
    data_info_repr += "Sample tensor name: the_samples"
    assert repr(data_info) == data_info_repr

    data_info = DataInfo(
        list_name="a_list", sample_name="the_samples", target_name="the_targets"
    )

    data_info_repr += "\nTarget tensor name: the_targets"

    assert repr(data_info) == data_info_repr

    data_info = DataInfo(
        list_name="a_list",
        sample_name="the_samples",
        target_name="the_targets",
        num_classes=23,
    )
    data_info_repr += "\nNumber of classes: 23"

    assert repr(data_info) == data_info_repr


@pytest.mark.skipif(
    not (shouldrun_torch or shouldrun_tf), reason="Requires TF or PyTorch"
)
def test_wrong_dataloaders(fileutils, wlmutils):
    test_dir = fileutils.make_test_dir()
    exp = Experiment("test-wrong-dataloaders", exp_path=test_dir, launcher=wlmutils.get_test_launcher())
    orc = wlmutils.get_orchestrator()
    exp.generate(orc)
    exp.start(orc)

    if shouldrun_tf:
        with pytest.raises(SSInternalError):
            _ = TFDataGenerator(
                data_info_or_list_name="test_data_list",
                address=orc.get_address()[0],
                cluster=False,
                max_fetch_trials=1,
            )
        with pytest.raises(TypeError):
            _ = TFStaticDataGenerator(
                test_data_info_repr=1,
                address=orc.get_address()[0],
                cluster=False,
                max_fetch_trials=1,
            )

    if shouldrun_torch:
        with pytest.raises(SSInternalError):
            torch_data_gen = TorchDataGenerator(
                data_info_or_list_name="test_data_list",
                address=orc.get_address()[0],
                cluster=False,
            )
            torch_data_gen.init_samples(init_trials=1)

    exp.stop(orc)
