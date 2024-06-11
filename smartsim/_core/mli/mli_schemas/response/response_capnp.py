"""This is an automatically generated stub for `response.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "response.capnp"))
Response = capnp.load(module_file).Response
ResponseBuilder = Response
ResponseReader = Response
