"""This is an automatically generated stub for `request.capnp`."""
import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "request.capnp"))
ChannelDescriptor = capnp.load(module_file).ChannelDescriptor
ChannelDescriptorBuilder = ChannelDescriptor
ChannelDescriptorReader = ChannelDescriptor
ModelKey = capnp.load(module_file).ModelKey
ModelKeyBuilder = ModelKey
ModelKeyReader = ModelKey
Request = capnp.load(module_file).Request
RequestBuilder = Request
RequestReader = Request
