@0x9a0aeb2e04838fb1;

using Enums = import "../enums/enums.capnp";

struct Tensor {
  blob @0 :Data;
  tensorDescriptor @1 :TensorDescriptor;
}

struct TensorDescriptor {
  dimensions @0 :List(Int32);
  order @1 :Enums.Order;
  dataType @2 :Enums.NumericalType;
}

struct TensorKey {
  key @0 :Text;
}