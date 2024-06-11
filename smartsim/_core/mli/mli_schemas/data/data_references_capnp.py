"""This is an automatically generated stub for `data_references.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "data_references.capnp"))
ModelKey = capnp.load(module_file).ModelKey
ModelKeyBuilder = ModelKey
ModelKeyReader = ModelKey
TensorKey = capnp.load(module_file).TensorKey
TensorKeyBuilder = TensorKey
TensorKeyReader = TensorKey
