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

import argparse
import contextlib
import io
import os
import os.path
import tempfile
import typing as t
from types import TracebackType

import numpy as np
from smartredis import Client

from smartsim import Experiment
from smartsim._core._cli.utils import SMART_LOGGER_FORMAT
from smartsim._core.types import Device
from smartsim._core.utils.helpers import installed_redisai_backends
from smartsim._core.utils.network import find_free_port
from smartsim.log import get_logger

logger = get_logger("Smart", fmt=SMART_LOGGER_FORMAT)

# Many of the functions in this module will import optional
# ML python packages only if they are needed to test the build is working
#
# pylint: disable=import-error,import-outside-toplevel
# mypy: disable-error-code="import"


if t.TYPE_CHECKING:
    from multiprocessing.connection import Connection

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


def execute(
    args: argparse.Namespace, _unparsed_args: t.Optional[t.List[str]] = None, /
) -> int:
    """Validate the SmartSim installation works as expected given a
    simple experiment
    """
    backends = installed_redisai_backends()
    temp_dir = ""
    device = Device(args.device)
    try:
        with contextlib.ExitStack() as ctx:
            temp_dir = ctx.enter_context(_VerificationTempDir(dir=os.getcwd()))
            validate_env = {
                "SR_LOG_LEVEL": os.environ.get("SR_LOG_LEVEL", "INFO"),
                "SR_LOG_FILE": os.environ.get(
                    "SR_LOG_FILE", os.path.join(temp_dir, "smartredis.log")
                ),
            }
            if device == Device.GPU:
                validate_env["CUDA_VISIBLE_DEVICES"] = "0"
            ctx.enter_context(_env_vars_set_to(validate_env))
            test_install(
                location=temp_dir,
                port=args.port,
                device=device,
                with_tf="tensorflow" in backends,
                with_pt="torch" in backends,
                with_onnx="onnxruntime" in backends,
            )
    except Exception as e:
        logger.error(
            "SmartSim failed to run a simple experiment!\n"
            f"Experiment failed due to the following exception:\n{e}",
            exc_info=True,
        )
        if temp_dir:
            logger.info(f"Output files are available at `{temp_dir}`")
        return os.EX_SOFTWARE
    return os.EX_OK


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Build the parser for the command"""
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=None,
        help=(
            "The port on which to run the orchestrator for the mini experiment. "
            "If not provided, `smart` will attempt to automatically select an "
            "open port"
        ),
    )
    parser.add_argument(
        "--device",
        type=str.lower,
        default=Device.CPU.value,
        choices=[device.value for device in Device],
        help="Device to test the ML backends against",
    )


def test_install(
    location: str,
    port: t.Optional[int],
    device: Device,
    with_tf: bool,
    with_pt: bool,
    with_onnx: bool,
) -> None:
    exp = Experiment("ValidationExperiment", exp_path=location, launcher="local")
    exp.telemetry.disable()
    port = find_free_port() if port is None else port

    with _make_managed_local_orc(exp, port) as client:
        logger.info("Verifying Tensor Transfer")
        client.put_tensor("plain-tensor", np.ones((1, 1, 3, 3)))
        client.get_tensor("plain-tensor")
        if with_pt:
            logger.info("Verifying Torch Backend")
            _test_torch_install(client, device)
        if with_onnx:
            logger.info("Verifying ONNX Backend")
            _test_onnx_install(client, device)
        if with_tf:  # Run last in case TF locks an entire GPU
            logger.info("Verifying TensorFlow Backend")
            _test_tf_install(client, location, device)
        logger.info("Success!")


@contextlib.contextmanager
def _env_vars_set_to(
    evars: t.Mapping[str, t.Optional[str]]
) -> t.Generator[None, None, None]:
    envvars = tuple((var, os.environ.pop(var, None), val) for var, val in evars.items())
    for var, _, tmpval in envvars:
        _set_or_del_env_var(var, tmpval)
    try:
        yield
    finally:
        for var, origval, _ in reversed(envvars):
            _set_or_del_env_var(var, origval)


def _set_or_del_env_var(var: str, val: t.Optional[str]) -> None:
    if val is not None:
        os.environ[var] = val
    else:
        os.environ.pop(var, None)


@contextlib.contextmanager
def _make_managed_local_orc(
    exp: Experiment, port: int
) -> t.Generator[Client, None, None]:
    """Context managed orc that will be stopped if an exception is raised"""
    orc = exp.create_database(db_nodes=1, interface="lo", port=port)
    exp.generate(orc)
    exp.start(orc)
    try:
        (client_addr,) = orc.get_address()
        yield Client(False, address=client_addr)
    finally:
        exp.stop(orc)


def _test_tf_install(client: Client, tmp_dir: str, device: Device) -> None:

    model_path, inputs, outputs = _build_tf_frozen_model(tmp_dir)

    client.set_model_from_file(
        "keras-fcn",
        model_path,
        "TF",
        device=device.value.upper(),
        inputs=inputs,
        outputs=outputs,
    )
    client.put_tensor("keras-input", np.random.rand(1, 28, 28).astype(np.float32))
    client.run_model("keras-fcn", inputs=["keras-input"], outputs=["keras-output"])
    client.get_tensor("keras-output")


def _build_tf_frozen_model(tmp_dir: str) -> t.Tuple[str, t.List[str], t.List[str]]:

    from tensorflow import keras  # pylint: disable=no-name-in-module

    from smartsim.ml.tf import freeze_model

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
    model_path, inputs, outputs = freeze_model(fcn, tmp_dir, "keras_model.pb")
    return model_path, inputs, outputs


def _test_torch_install(client: Client, device: Device) -> None:
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv: t.Callable[..., torch.Tensor] = nn.Conv2d(1, 1, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    if device == Device.GPU:
        device_ = torch.device("cuda")
    else:
        device_ = torch.device("cpu")

    net = Net()
    net.to(device_)
    net.eval()

    forward_input = torch.rand(1, 1, 3, 3).to(device_)
    traced = torch.jit.trace(  # type: ignore[no-untyped-call, unused-ignore]
        net, forward_input
    )

    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)  # type: ignore[no-untyped-call, unused-ignore]
    model = buffer.getvalue()

    client.set_model("torch-nn", model, backend="TORCH", device=device.value.upper())
    client.put_tensor("torch-in", torch.rand(1, 1, 3, 3).numpy())
    client.run_model("torch-nn", inputs=["torch-in"], outputs=["torch-out"])
    client.get_tensor("torch-out")


def _test_onnx_install(client: Client, device: Device) -> None:
    from skl2onnx import to_onnx
    from sklearn.cluster import KMeans

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    model = KMeans(n_clusters=2, n_init=10)
    model.fit(data)

    kmeans = to_onnx(model, data, target_opset=11)
    model = kmeans.SerializeToString()
    sample = np.arange(20, dtype=np.float32).reshape(10, 2)

    client.put_tensor("onnx-input", sample)
    client.set_model("onnx-kmeans", model, "ONNX", device=device.value.upper())
    client.run_model(
        "onnx-kmeans", inputs=["onnx-input"], outputs=["onnx-labels", "onnx-transform"]
    )
    client.get_tensor("onnx-labels")
