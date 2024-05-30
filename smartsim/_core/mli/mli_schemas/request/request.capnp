@0xa27f0152c7bb299e;

using Enums = import "../enums/enums.capnp";
using Tensors = import "../tensor/tensor.capnp";
using RequestAttributes = import "request_attributes/request_attributes.capnp";

struct ChannelDescriptor {
  reply @0 :Data;
}

struct ModelKey {
  key @0 :Text;
}

struct Request {
  replyChannel @0 :ChannelDescriptor;
  model :union {
    modelKey @1 :ModelKey;
    modelData @2 :Data;
  }
  device :union {
    deviceType @3 :Enums.Device;
    noDevice @4 :Void;
  }
  input :union {
    inputKeys @5 :List(Tensors.TensorKey);
    inputData @6 :List(Tensors.Tensor);
  }
  output :union {
    outputKeys @7 :List(Tensors.TensorKey);
    outputData @8 :List(Tensors.Tensor);
  }
  customAttributes :union {
    torchCNN @9 :RequestAttributes.TorchRequestAttributes;
    tfCNN @10 :RequestAttributes.TensorflowRequestAttributes;
    none @11 :Void;
  }
}