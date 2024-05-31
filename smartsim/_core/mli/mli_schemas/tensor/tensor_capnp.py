"""This is an automatically generated stub for `tensor.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "tensor.capnp"))
Tensor = capnp.load(module_file).Tensor
TensorBuilder = Tensor
TensorReader = Tensor
TensorDescriptor = capnp.load(module_file).TensorDescriptor
TensorDescriptorBuilder = TensorDescriptor
TensorDescriptorReader = TensorDescriptor
TensorKey = capnp.load(module_file).TensorKey
TensorKeyBuilder = TensorKey
TensorKeyReader = TensorKey
