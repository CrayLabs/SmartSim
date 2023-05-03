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


from pathlib import Path

from .._core.utils.helpers import init_default

__all__ = ["DBObject", "DBModel", "DBScript"]


class DBObject:
    """Base class for ML objects residing on DB. Should not
    be instantiated.
    """

    def __init__(self, name, func, file_path, device, devices_per_node):
        self.name = name
        self.func = func
        if file_path:
            self.file = self._check_filepath(file_path)
        else:
            # Need to have this explicitly to check on it
            self.file = None
        self.device = self._check_device(device)
        self.devices_per_node = devices_per_node

    @property
    def is_file(self):
        if self.func:
            return False
        return True

    @staticmethod
    def _check_tensor_args(inputs, outputs):
        inputs = init_default([], inputs, (list, str))
        outputs = init_default([], outputs, (list, str))
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return inputs, outputs

    @staticmethod
    def _check_backend(backend):
        backend = backend.upper()
        all_backends = ["TF", "TORCH", "ONNX"]
        if backend in all_backends:
            return backend
        else:
            raise ValueError(
                f"Backend type {backend} unsupported. Options are {all_backends}"
            )

    @staticmethod
    def _check_filepath(file):
        file_path = Path(file).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        return file_path

    @staticmethod
    def _check_device(device):
        device = device.upper()
        if not device.startswith("CPU") and not device.startswith("GPU"):
            raise ValueError("Device argument must start with either CPU or GPU")
        return device

    def _enumerate_devices(self):
        """Enumerate devices for a DBObject

        :param dbobject: DBObject to enumerate
        :type dbobject: DBObject
        :return: list of device names
        :rtype: list[str]
        """
        devices = []
        if ":" in self.device and self.devices_per_node > 1:
            msg = "Cannot set devices_per_node>1 if a device numeral is specified, "
            msg += f"the device was set to {self.device} and devices_per_node=={self.devices_per_node}"
            raise ValueError(msg)
        if self.device in ["CPU", "GPU"] and self.devices_per_node > 1:
            for device_num in range(self.devices_per_node):
                devices.append(f"{self.device}:{str(device_num)}")
        else:
            devices = [self.device]

        return devices


class DBScript(DBObject):
    def __init__(
        self, name, script=None, script_path=None, device="CPU", devices_per_node=1
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
    def script(self):
        return self.func

    def __str__(self):
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
        name,
        backend,
        model=None,
        model_file=None,
        device="CPU",
        devices_per_node=1,
        batch_size=0,
        min_batch_size=0,
        min_batch_timeout=0,
        tag="",
        inputs=None,
        outputs=None,
    ):
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
    def model(self):
        return self.func

    def __str__(self):
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
