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

import typing as t

from pathlib import Path
from .._core.utils import init_default
from ..error import SSUnsupportedError


__all__ = ["DBObject", "DBModel", "DBScript"]


class DBObject:
    """Base class for ML objects residing on DB. Should not
    be instantiated.
    """

    def __init__(
        self,
        name: str,
        func: t.Optional[str],
        file_path: t.Optional[str],
        device: t.Literal["CPU", "GPU"],
        devices_per_node: int,
    ) -> None:
        self.name = name
        self.func = func
        self.file: t.Optional[
            Path
        ] = None  # Need to have this explicitly to check on it
        if file_path:
            self.file = self._check_filepath(file_path)
        self.device = self._check_device(device)
        self.devices_per_node = devices_per_node
        self._check_devices(device, devices_per_node)

    @property
    def devices(self) -> t.List[str]:
        return self._enumerate_devices()

    @property
    def is_file(self) -> bool:
        if self.func:
            return False
        return True

    @staticmethod
    def _check_tensor_args(
        inputs: t.Union[str, t.Optional[t.List[str]]],
        outputs: t.Union[str, t.Optional[t.List[str]]],
    ) -> t.Tuple[t.List[str], t.List[str]]:
        inputs = init_default([], inputs, (list, str))
        outputs = init_default([], outputs, (list, str))

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
    def _check_device(device: t.Literal["CPU", "GPU"]) -> str:
        device = t.cast(t.Literal["CPU", "GPU"], device.upper())
        if not device.startswith("CPU") and not device.startswith("GPU"):
            raise ValueError("Device argument must start with either CPU or GPU")
        return device

    def _enumerate_devices(self) -> t.List[str]:
        """Enumerate devices for a DBObject

        :param dbobject: DBObject to enumerate
        :type dbobject: DBObject
        :return: list of device names
        :rtype: list[str]
        """

        if self.device == "GPU" and self.devices_per_node > 1:
            return [
                f"{self.device}:{str(device_num)}"
                for device_num in range(self.devices_per_node)
            ]

        return [self.device]

    @staticmethod
    def _check_devices(
        device: t.Literal["CPU", "GPU"], devices_per_node: int
    ) -> None:
        if devices_per_node == 1:
            return

        if ":" in device:
            msg = "Cannot set devices_per_node>1 if a device numeral is specified, "
            msg += f"the device was set to {device} and \
                devices_per_node=={devices_per_node}"
            raise ValueError(msg)
        if device == "CPU":
            raise SSUnsupportedError(
                "Cannot set devices_per_node>1 if CPU is specified under devices"
            )


class DBScript(DBObject):
    def __init__(
        self,
        name: str,
        script: t.Optional[str] = None,
        script_path: t.Optional[str] = None,
        device: t.Literal["CPU", "GPU"] = "CPU",
        devices_per_node: int = 1,
    ):
        """TorchScript code represenation

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the model being stored on the first N devices of type ``device``.

        One of either script (in memory representation) or script_path (file)
        must be provided

        :param name: key to store script under
        :type name: str
        :param script: TorchScript code
        :type script: str, optional
        :param script_path: path to TorchScript code, defaults to None
        :type script_path: str, optional
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: number of devices to store the script on
        :type devices_per_node: int
        """
        super().__init__(name, script, script_path, device, devices_per_node)
        if not script and not script_path:
            raise ValueError("Either script or script_path must be provided")

    @property
    def script(self) -> t.Optional[str]:
        return self.func

    def __str__(self) -> str:
        desc_str = "Name: " + self.name + "\n"
        if self.func:
            desc_str += "Func: " + self.func + "\n"
        if self.file:
            desc_str += "File path: " + str(self.file) + "\n"
        devices_str = self.device + (
            "s per node\n" if self.devices_per_node > 1 else " per node\n"
        )
        desc_str += "Devices: " + str(self.devices_per_node) + " " + devices_str
        return desc_str


class DBModel(DBObject):
    def __init__(
        self,
        name: str,
        backend: str,
        model: t.Optional[str] = None,
        model_file: t.Optional[str] = None,
        device: t.Literal["CPU", "GPU"] = "CPU",
        devices_per_node: int = 1,
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
        :type name: str
        :param model: model in memory
        :type model: str, optional
        :param model_file: serialized model
        :type model_file: file path to model, optional
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param device: name of device for execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: number of devices to store the model on
        :type devices_per_node: int
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
        :param min_batch_timeout: time to wait for minimum batch size, defaults to 0
        :type min_batch_timeout: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: list[str], optional
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: list[str], optional
        """
        super().__init__(name, model, model_file, device, devices_per_node)
        self.backend = self._check_backend(backend)
        if not model and not model_file:
            raise ValueError("Either model or model_file must be provided")
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.min_batch_timeout = min_batch_timeout
        self.tag = tag
        self.inputs, self.outputs = self._check_tensor_args(inputs, outputs)

    @property
    def model(self) -> t.Union[str, None]:
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
