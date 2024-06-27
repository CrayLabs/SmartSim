"""This is an automatically generated stub for `model.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "model.capnp"))
Model = capnp.load(module_file).Model
ModelBuilder = Model
ModelReader = Model
