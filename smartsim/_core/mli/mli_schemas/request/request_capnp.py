"""This is an automatically generated stub for `request.capnp`."""

import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "request.capnp"))
ChannelDescriptor = capnp.load(module_file).ChannelDescriptor
ChannelDescriptorBuilder = ChannelDescriptor
ChannelDescriptorReader = ChannelDescriptor
Request = capnp.load(module_file).Request
RequestBuilder = Request
RequestReader = Request
