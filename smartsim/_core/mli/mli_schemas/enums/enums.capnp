@0xbf10df09fc39f95f;

enum Order {
  c @0; # row major (contiguous layout)
  f @1; # column major (fortran contiguous layout)
}

enum Device {
  cpu @0;
  gpu @1;
}

enum NumericalType {
  int8 @0;
  int16 @1;
  int32 @2;
  int64 @3;
  uInt8 @4;
  uInt16 @5;
  uInt32 @6;
  uInt64 @7;
  float32 @8; 
  float64 @9;
}

enum TorchTensorType {
  nested @0; # ragged
  sparse @1;
  tensor @2; # "normal" tensor
}

enum TFTensorType {
  ragged @0;
  sparse @1;
  variable @2;
  constant @3;
}