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

import io
import argparse
import tempfile
from pathlib import Path
from contextlib import contextmanager, ExitStack
from types import TracebackType
import typing as t

import numpy as np

from smartsim import Experiment
from smartredis import Client
from smartsim._core.utils.helpers import installed_redisai_backends

from smartsim.log import get_logger
from smartsim._core._cli.utils import smart_logger_format

logger = get_logger("Smart", fmt=smart_logger_format)


class VerificationTempDir(tempfile.TemporaryDirectory):
    def __exit__(
        self,
        exc: t.Optional[t.Type[BaseException]],
        value: t.Optional[BaseException],
        tb: t.Optional[TracebackType],
    ) -> None:
        """Only clean up the temp dir if no error"""
        if not value:
            super().__exit__(exc, value, tb)


def execute(_args: argparse.Namespace, /) -> int:
    backends = installed_redisai_backends()
    try:
        verify_install(
            with_tf="tensorflow" in backends,
            with_pt="torch" in backends,
            with_onnx="onnxruntime" in backends,
        )
        return 0
    except Exception as e:
        logger.error(
            "SmartSim failed to run a simple experiment. "
            f"Experiment failed do to the following exception:\n{e}"
        )
        return 2


def verify_install(with_tf: bool, with_pt: bool, with_onnx: bool) -> None:
    with ExitStack() as ctx:
        temp_dir = ctx.enter_context(VerificationTempDir())
        exp = Experiment(
            "VerificationExperiment", exp_path=str(temp_dir), launcher="local"
        )
        client = ctx.enter_context(_make_managed_orc(exp))
        client.put_tensor("plain-tensor", np.ones((1, 1, 3, 3)))
        client.get_tensor("plain-tensor")
        if with_tf:
            logger.info("Verifying TensorFlow Backend")
            _verify_tf_install(client)
        if with_pt:
            logger.info("Verifying Torch Backend")
            _verify_torch_install(client)
        if with_onnx:
            logger.info("Verifying ONNX Backend")
            _verify_onnx_install(client)


@contextmanager
def _make_managed_orc(exp: Experiment) -> t.Generator[Client, None, None]:
    orc = exp.create_database(db_nodes=1, port=8934, interface="lo")
    exp.generate(orc)
    exp.start(orc)
    (client_addr,) = orc.get_address()
    client = Client(address=client_addr, cluster=False)
    try:
        yield client
    finally:
        exp.stop(orc)


def _verify_tf_install(client: Client) -> None:
    import tensorflow as tf
    from tensorflow import keras

    from smartsim.ml.tf import serialize_model

    fcn = keras.Sequential(
        layers=[
            keras.layers.InputLayer(input_shape=(28, 28), name="input"),
            keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense"),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="FullyConnectedNetwork",
    )
    fcn.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model, inputs, outputs = serialize_model(fcn)

    client.set_model(
        "keras-fcn", model, "TF", device="CPU", inputs=inputs, outputs=outputs
    )
    client.put_tensor("keras-input", np.random.rand(1, 28, 28).astype(np.float32))
    client.run_model("keras-fcn", inputs=["keras-input"], outputs=["keras-output"])
    client.get_tensor("keras-output")


def _verify_torch_install(client: Client) -> None:
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x: t.Any) -> None:
            return self.conv(x)

    net = Net()
    forward_input = torch.rand(1, 1, 3, 3)
    traced = torch.jit.trace(net, forward_input)  # type: ignore[no-untyped-call]
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)  # type: ignore[no-untyped-call]
    model = buffer.getvalue()

    client.set_model("torch-nn", model, backend="TORCH", device="CPU")
    client.put_tensor("torch-in", torch.rand(1, 1, 3, 3).numpy())
    client.run_model("torch-nn", inputs=["torch-in"], outputs=["torch-out"])
    client.get_tensor("torch-out")


def _verify_onnx_install(client: Client) -> None:
    from skl2onnx import to_onnx  # type: ignore[import]
    from sklearn.cluster import KMeans  # type: ignore[import]

    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    model = KMeans(n_clusters=2)
    model.fit(X)

    kmeans = to_onnx(model, X, target_opset=11)
    model = kmeans.SerializeToString()
    sample = np.arange(20, dtype=np.float32).reshape(10, 2)

    client.put_tensor("onnx-input", sample)
    client.set_model("onnx-kmeans", model, "ONNX", device="CPU")
    client.run_model(
        "onnx-kmeans", inputs=["onnx-input"], outputs=["onnx-labels", "onnx-transform"]
    )
    client.get_tensor("onnx-labels")
