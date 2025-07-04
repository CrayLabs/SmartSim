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

import typing as t
from pathlib import Path

from smartsim._core.types import Device

from ..error import SSUnsupportedError

__all__ = ["DBObject", "DBModel", "DBScript"]


_DBObjectFuncT = t.TypeVar("_DBObjectFuncT", str, bytes)


class DBObject(t.Generic[_DBObjectFuncT]):
    """Base class for ML objects residing on DB. Should not
    be instantiated.
    """

    def __init__(
        self,
        name: str,
        func: t.Optional[_DBObjectFuncT],
        file_path: t.Optional[str],
        device: str,
        devices_per_node: int,
        first_device: int,
    ) -> None:
        self.name = name
        self.func: t.Optional[_DBObjectFuncT] = func
        self.file: t.Optional[Path] = (
            None  # Need to have this explicitly to check on it
        )
        if file_path:
            self.file = self._check_filepath(file_path)
        self.device = self._check_device(device)
        self.devices_per_node = devices_per_node
        self.first_device = first_device
        self._check_devices(device, devices_per_node, first_device)

    @property
    def devices(self) -> t.List[str]:
        return self._enumerate_devices()

    @property
    def is_file(self) -> bool:
        return not self.func

    @staticmethod
    def _check_tensor_args(
        inputs: t.Union[str, t.Optional[t.List[str]]],
        outputs: t.Union[str, t.Optional[t.List[str]]],
    ) -> t.Tuple[t.List[str], t.List[str]]:
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return inputs or [], outputs or []

    @staticmethod
    def _check_backend(backend: str) -> str:
        backend = backend.upper()
        all_backends = ["TF", "TORCH", "ONNX"]
        if backend in all_backends:
            return backend

        raise ValueError(
            f"Backend type {backend} unsupported. Options are {all_backends}"
        )

    @staticmethod
    def _check_filepath(file: str) -> Path:
        file_path = Path(file).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        return file_path

    @staticmethod
    def _check_device(device: str) -> str:
        valid_devices = [Device.CPU.value, Device.GPU.value]
        if not any(device.lower().startswith(dev) for dev in valid_devices):
            raise ValueError("Device argument must start with either CPU or GPU")
        return device

    def _enumerate_devices(self) -> t.List[str]:
        """Enumerate devices for a DBObject

        :param dbobject: DBObject to enumerate
        :return: list of device names
        """

        if self.device == "GPU" and self.devices_per_node > 1:
            return [
                f"{self.device}:{device_num}"
                for device_num in range(
                    self.first_device, self.first_device + self.devices_per_node
                )
            ]

        return [self.device]

    @staticmethod
    def _check_devices(
        device: str,
        devices_per_node: int,
        first_device: int,
    ) -> None:
        if device.lower() == Device.CPU.value and devices_per_node > 1:
            raise SSUnsupportedError(
                "Cannot set devices_per_node>1 if CPU is specified under devices"
            )

        if device.lower() == Device.CPU.value and first_device > 0:
            raise SSUnsupportedError(
                "Cannot set first_device>0 if CPU is specified under devices"
            )

        if devices_per_node == 1:
            return

        if ":" in device:
            msg = "Cannot set devices_per_node>1 if a device numeral is specified, "
            msg += f"the device was set to {device} and \
                devices_per_node=={devices_per_node}"
            raise ValueError(msg)


class DBScript(DBObject[str]):
    def __init__(
        self,
        name: str,
        script: t.Optional[str] = None,
        script_path: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
    ):
        """TorchScript code represenation

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the script being stored on the first N devices of type ``device``;
        additionally setting ``first_device=M`` will instead result in the
        script being stored on devices M through M + N -1.

        One of either script (in memory representation) or script_path (file)
        must be provided

        :param name: key to store script under
        :param script: TorchScript code
        :param script_path: path to TorchScript code
        :param device: device for script execution
        :param devices_per_node: number of devices to store the script on
        :param first_device: first devices to store the script on
        """
        super().__init__(
            name, script, script_path, device, devices_per_node, first_device
        )
        if not script and not script_path:
            raise ValueError("Either script or script_path must be provided")

    @property
    def script(self) -> t.Optional[t.Union[bytes, str]]:
        return self.func

    def __str__(self) -> str:
        desc_str = "Name: " + self.name + "\n"
        if self.func:
            desc_str += "Func: " + str(self.func) + "\n"
        if self.file:
            desc_str += "File path: " + str(self.file) + "\n"
        devices_str = self.device + (
            "s per node\n" if self.devices_per_node > 1 else " per node\n"
        )
        desc_str += "Devices: " + str(self.devices_per_node) + " " + devices_str
        if self.first_device > 0:
            desc_str += "First device: " + str(self.first_device) + "\n"
        return desc_str


class DBModel(DBObject[bytes]):
    def __init__(
        self,
        name: str,
        backend: str,
        model: t.Optional[bytes] = None,
        model_file: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
        batch_size: int = 0,
        min_batch_size: int = 0,
        min_batch_timeout: int = 0,
        tag: str = "",
        inputs: t.Optional[t.List[str]] = None,
        outputs: t.Optional[t.List[str]] = None,
    ) -> None:
        """A TF, TF-lite, PT, or ONNX model to load into the DB at runtime

        One of either model (in memory representation) or model_path (file)
        must be provided

        :param name: key to store model under
        :param model: model in memory
        :param model_file: serialized model
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :param device: name of device for execution
        :param devices_per_node: number of devices to store the model on
        :param first_device: The first device to store the model on
        :param batch_size: batch size for execution
        :param min_batch_size: minimum batch size for model execution
        :param min_batch_timeout: time to wait for minimum batch size
        :param tag: additional tag for model information
        :param inputs: model inputs (TF only)
        :param outputs: model outupts (TF only)
        """
        super().__init__(
            name, model, model_file, device, devices_per_node, first_device
        )
        self.backend = self._check_backend(backend)
        if not model and not model_file:
            raise ValueError("Either model or model_file must be provided")
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.min_batch_timeout = min_batch_timeout
        self.tag = tag
        self.inputs, self.outputs = self._check_tensor_args(inputs, outputs)

    @property
    def model(self) -> t.Optional[bytes]:
        return self.func

    def __str__(self) -> str:
        desc_str = "Name: " + self.name + "\n"
        if self.model:
            desc_str += "Model stored in memory\n"
        if self.file:
            desc_str += "File path: " + str(self.file) + "\n"
        devices_str = self.device + (
            "s per node\n" if self.devices_per_node > 1 else " per node\n"
        )
        desc_str += "Devices: " + str(self.devices_per_node) + " " + devices_str
        if self.first_device > 0:
            desc_str += "First_device: " + str(self.first_device) + "\n"
        desc_str += "Backend: " + str(self.backend) + "\n"
        if self.batch_size:
            desc_str += "Batch size: " + str(self.batch_size) + "\n"
        if self.min_batch_size:
            desc_str += "Min batch size: " + str(self.min_batch_size) + "\n"
        if self.min_batch_timeout:
            desc_str += "Min batch time out: " + str(self.min_batch_timeout) + "\n"
        if self.tag:
            desc_str += "Tag: " + self.tag + "\n"
        if self.inputs:
            desc_str += "Inputs: " + str(self.inputs) + "\n"
        if self.outputs:
            desc_str += "Outputs: " + str(self.outputs) + "\n"
        return desc_str
