@0xa05dcb4444780705;

using Tensors = import "../tensor/tensor.capnp";
using ResponseAttributes = import "response_attributes/response_attributes.capnp";

struct Response {
  status @0 :Int32;
  statusMessage @1 :Text;
  result :union {
    keys @2 :List(Tensors.TensorKey);
    data @3 :List(Tensors.Tensor); # do we need response attributes for each worker type?
  }
  customAttributes :union {
    torchCNN @4 :ResponseAttributes.TorchResponseAttributes;
    tfCNN @5 :ResponseAttributes.TensorflowResponseAttributes;
    none @6 :Void;
  }
}