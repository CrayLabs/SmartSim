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
import socket
from types import TracebackType

import numpy as np
from smartredis import Client

from smartsim import Experiment
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

# Many of the functions in this module will import optional
# ML python packages only if they are needed to test the build is working
#
# pylint: disable=import-error,import-outside-toplevel
# mypy: disable-error-code="import"


if t.TYPE_CHECKING:
    # Pylint disable needed for old version of pylint w/ TF 2.6.2
    # pylint: disable-next=unsubscriptable-object
    _TemporaryDirectory = tempfile.TemporaryDirectory[str]
else:
    _TemporaryDirectory = tempfile.TemporaryDirectory


class _VerificationTempDir(_TemporaryDirectory):
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
            test_install(
                location=temp_dir,
                port=None,  # TODO: allow users to pass as arg `--port`?
                with_tf="tensorflow" in backends,
                with_pt="torch" in backends,
                with_onnx="onnxruntime" in backends,
            )
    except Exception as e:
        logger.error(
            "SmartSim failed to run a simple experiment!\n"
            f"Experiment failed due to the following exception:\n{e}\n\n"
            f"Output files are available at `{temp_dir}`"
        )
        return 2
    return 0


def test_install(
    location: str,
    port: t.Optional[int],
    with_tf: bool,
    with_pt: bool,
    with_onnx: bool,
) -> None:
    exp = Experiment("TestExperiment", exp_path=location, launcher="local")
    port = port or _find_free_port()
    with _make_managed_local_orc(exp, port) as client:
        logger.info("Verifying Tensor Transfer")
        client.put_tensor("plain-tensor", np.ones((1, 1, 3, 3)))
        client.get_tensor("plain-tensor")
        if with_tf:
            logger.info("Verifying TensorFlow Backend")
            _test_tf_install(client)
        if with_pt:
            logger.info("Verifying Torch Backend")
            _test_torch_install(client)
        if with_onnx:
            logger.info("Verifying ONNX Backend")
            _test_onnx_install(client)


@contextmanager
def _make_managed_local_orc(
    exp: Experiment, port: int
) -> t.Generator[Client, None, None]:
    """Context managed orc that will be stopped if an exception is raised"""
    orc = exp.create_database(db_nodes=1, interface="lo", port=port)
    exp.generate(orc)
    exp.start(orc)
    try:
        (client_addr,) = orc.get_address()
        yield Client(address=client_addr, cluster=False)
    finally:
        exp.stop(orc)


def _find_free_port() -> int:
    """A 'good enough' way to find an open port to bind to"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        _, port = sock.getsockname()
        return t.cast(int, port)


def _test_tf_install(client: Client) -> None:
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


def _test_torch_install(client: Client) -> None:
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv: t.Callable[..., torch.Tensor] = nn.Conv2d(1, 1, 3)

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


def _test_onnx_install(client: Client) -> None:
    from skl2onnx import to_onnx
    from sklearn.cluster import KMeans

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
