"""This is an automatically generated stub for `response_attributes.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "response_attributes.capnp"))
TorchResponseAttributes = capnp.load(module_file).TorchResponseAttributes
TorchResponseAttributesBuilder = TorchResponseAttributes
TorchResponseAttributesReader = TorchResponseAttributes
TensorFlowResponseAttributes = capnp.load(module_file).TensorFlowResponseAttributes
TensorFlowResponseAttributesBuilder = TensorFlowResponseAttributes
TensorFlowResponseAttributesReader = TensorFlowResponseAttributes
