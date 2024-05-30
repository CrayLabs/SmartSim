@0xdd14d8ba5c06743f;

using Enums = import "../../enums/enums.capnp";

struct TorchRequestAttributes {
  tensorType @0 :Enums.TorchTensorType;
}

struct TensorflowRequestAttributes {
  name @0 :Text;
  tensorType @1 :Enums.TFTensorType;
}