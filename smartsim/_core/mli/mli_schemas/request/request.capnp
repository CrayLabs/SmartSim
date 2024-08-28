# BSD 2-Clause License

# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

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

@0xa27f0152c7bb299e;

using Tensors = import "../tensor/tensor.capnp";
using RequestAttributes = import "request_attributes/request_attributes.capnp";
using DataRef = import "../data/data_references.capnp";
using Models = import "../model/model.capnp";

struct ChannelDescriptor {
  descriptor @0 :Text;
}

struct Request {
  replyChannel @0 :ChannelDescriptor;
  model :union {
    key @1 :DataRef.FeatureStoreKey;
    data @2 :Models.Model;
  }
  input :union {
    keys @3 :List(DataRef.FeatureStoreKey);
    descriptors @4 :List(Tensors.TensorDescriptor);
  }
  output @5 :List(DataRef.FeatureStoreKey);
  outputDescriptors @6 :List(Tensors.OutputDescriptor);
  customAttributes :union {
    torch @7 :RequestAttributes.TorchRequestAttributes;
    tf @8 :RequestAttributes.TensorFlowRequestAttributes;
    none @9 :Void;
  }
}
