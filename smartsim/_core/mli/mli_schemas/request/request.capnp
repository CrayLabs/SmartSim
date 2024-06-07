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

enum Device {
  cpu @0;
  gpu @1;
}

struct ChannelDescriptor {
  reply @0 :Data;
}

struct Request {
  replyChannel @0 :ChannelDescriptor;
  model :union {
    modelKey @1 :DataRef.ModelKey;
    modelData @2 :Data;
  }
  device :union {
    deviceType @3 :Device;
    noDevice @4 :Void;
  }
  input :union {
    inputKeys @5 :List(DataRef.TensorKey);
    inputData @6 :List(Tensors.Tensor);
  }
  output :union {
    outputKeys @7 :List(DataRef.TensorKey);
    outputData @8 :Void;
  }
  outputOptions @9 :List(Tensors.OutputTensorDescriptor);
  customAttributes :union {
    torch @10 :RequestAttributes.TorchRequestAttributes;
    tf @11 :RequestAttributes.TensorFlowRequestAttributes;
    none @12 :Void;
  }
}