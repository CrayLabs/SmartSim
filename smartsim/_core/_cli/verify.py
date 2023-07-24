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

import argparse
import io
import tempfile
import typing as t
from contextlib import contextmanager
from types import TracebackType

import numpy as np
from smartredis import Client

from smartsim import Experiment
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim.log import get_logger

SMART_LOGGER_FORMAT = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)  # TODO: This

# Many of the functions in this module will import optional
# ml python packages only if they are needed to verify the build is working
# pylint: disable=import-outside-toplevel

class _VerificationTempDir(tempfile.TemporaryDirectory):
    """A Temporary directory to be used as a context manager that will only
    clean itself up if no error is raised within its context
    """

    def __exit__(
        self,
        exc: t.Optional[t.Type[BaseException]],
        value: t.Optional[BaseException],
        tb: t.Optional[TracebackType],
    ) -> None:
        if not value:  # Yay, no error! Clean up as normal
            super().__exit__(exc, value, tb)
        else:  # Uh-oh! Better make sure this is not implicitly cleaned up
            self._finalizer.detach()  # type: ignore[attr-defined]


def execute(_args: argparse.Namespace, /) -> int:
    backends = installed_redisai_backends()
    try:
        with _VerificationTempDir() as temp_dir:
            verify_install(
                location=temp_dir,
                with_tf="tensorflow" in backends,
                with_pt="torch" in backends,
                with_onnx="onnxruntime" in backends,
            )
    except Exception as e:
        logger.error(
            "SmartSim failed to run a simple experiment!\n"
            f"Experiment failed do to the following exception:\n{e}\n\n"
            f"Output files are available at `{temp_dir}`"
        )
        return 2
    return 0


def verify_install(
    location: str, with_tf: bool, with_pt: bool, with_onnx: bool
) -> None:
    exp = Experiment("Verification", exp_path=location, launcher="local")
    with _make_managed_orc(exp) as client:
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
    try:
        (client_addr,) = orc.get_address()
        yield Client(address=client_addr, cluster=False)
    finally:
        exp.stop(orc)


def _verify_tf_install(client: Client) -> None:
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
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    # These imports will fail typecheck unless built with the `--onnx` flag,
    # which is not available for py3.10 at the moment
    from skl2onnx import to_onnx  # type: ignore[import]
    from sklearn.cluster import KMeans  # type: ignore[import]

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    model = KMeans(n_clusters=2)
    model.fit(data)

    kmeans = to_onnx(model, data, target_opset=11)
    model = kmeans.SerializeToString()
    sample = np.arange(20, dtype=np.float32).reshape(10, 2)

    client.put_tensor("onnx-input", sample)
    client.set_model("onnx-kmeans", model, "ONNX", device="CPU")
    client.run_model(
        "onnx-kmeans", inputs=["onnx-input"], outputs=["onnx-labels", "onnx-transform"]
    )
    client.get_tensor("onnx-labels")
