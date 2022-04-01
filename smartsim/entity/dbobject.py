from pathlib import Path
from .._core.utils.helpers import init_default


class DBObject:
    def __init__(self, name, func, file_path, device, devices_per_node):
        self.name = name
        self.func = func
        if file_path:
            self.file = self._check_filepath(file_path)
        self.device = self._check_device(device)
        self.devices_per_node = devices_per_node

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
                f"Backend type {backend} unsupported. Options are {all_backends}")

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

class DBScript(DBObject):

    def __init__(self,
                 name,
                 script=None,
                 script_path=None,
                 device="CPU",
                 devices_per_node=1
                ):
        """TorchScript code represenation

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the model being stored in the first N devices of type ``device``.

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
        """
        super().__init__(name, script, script_path, device, devices_per_node)
        if not script and not script_path:
            raise ValueError("Either script or script_path must be provided")

    @property
    def script(self):
        return self.func

class DBModel(DBObject):
    def __init__(self,
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
                 outputs=None):
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
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
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

