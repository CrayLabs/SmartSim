"""This is an automatically generated stub for `enums.capnp`."""
from __future__ import annotations

from typing import Literal

Order = Literal["c", "f"]
Device = Literal["cpu", "gpu"]
NumericalType = Literal["int8", "int16", "int32", "int64", "uInt8", "uInt16", "uInt32", "uInt64", "float32", "float64"]
TorchTensorType = Literal["nested", "sparse", "tensor"]
TFTensorType = Literal["ragged", "sparse", "variable", "constant"]
