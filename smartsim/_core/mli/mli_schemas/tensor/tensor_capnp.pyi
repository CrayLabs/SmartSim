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

"""This is an automatically generated stub for `tensor.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, Sequence

from ..data.data_references_capnp import (
    FeatureStoreKey,
    FeatureStoreKeyBuilder,
    FeatureStoreKeyReader,
)

Order = Literal["c", "f"]
NumericalType = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uInt8",
    "uInt16",
    "uInt32",
    "uInt64",
    "float32",
    "float64",
]
ReturnNumericalType = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uInt8",
    "uInt16",
    "uInt32",
    "uInt64",
    "float32",
    "float64",
    "none",
    "auto",
]

class TensorDescriptor:
    dimensions: Sequence[int]
    order: Order
    dataType: NumericalType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TensorDescriptorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TensorDescriptorReader: ...
    @staticmethod
    def new_message() -> TensorDescriptorBuilder: ...
    def to_dict(self) -> dict: ...

class TensorDescriptorReader(TensorDescriptor):
    def as_builder(self) -> TensorDescriptorBuilder: ...

class TensorDescriptorBuilder(TensorDescriptor):
    @staticmethod
    def from_dict(dictionary: dict) -> TensorDescriptorBuilder: ...
    def copy(self) -> TensorDescriptorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TensorDescriptorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class OutputDescriptor:
    order: Order
    optionalKeys: Sequence[
        FeatureStoreKey | FeatureStoreKeyBuilder | FeatureStoreKeyReader
    ]
    optionalDimension: Sequence[int]
    optionalDatatype: ReturnNumericalType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[OutputDescriptorReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> OutputDescriptorReader: ...
    @staticmethod
    def new_message() -> OutputDescriptorBuilder: ...
    def to_dict(self) -> dict: ...

class OutputDescriptorReader(OutputDescriptor):
    optionalKeys: Sequence[FeatureStoreKeyReader]
    def as_builder(self) -> OutputDescriptorBuilder: ...

class OutputDescriptorBuilder(OutputDescriptor):
    optionalKeys: Sequence[
        FeatureStoreKey | FeatureStoreKeyBuilder | FeatureStoreKeyReader
    ]
    @staticmethod
    def from_dict(dictionary: dict) -> OutputDescriptorBuilder: ...
    def copy(self) -> OutputDescriptorBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> OutputDescriptorReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
