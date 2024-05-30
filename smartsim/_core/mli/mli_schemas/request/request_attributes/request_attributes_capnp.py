"""This is an automatically generated stub for `request_attributes.capnp`."""
import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "request_attributes.capnp"))
TorchRequestAttributes = capnp.load(module_file).TorchRequestAttributes
TorchRequestAttributesBuilder = TorchRequestAttributes
TorchRequestAttributesReader = TorchRequestAttributes
TensorflowRequestAttributes = capnp.load(module_file).TensorflowRequestAttributes
TensorflowRequestAttributesBuilder = TensorflowRequestAttributes
TensorflowRequestAttributesReader = TensorflowRequestAttributes
