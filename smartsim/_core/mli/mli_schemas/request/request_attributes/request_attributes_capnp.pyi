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

"""This is an automatically generated stub for `request_attributes.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal

TorchTensorType = Literal["nested", "sparse", "tensor"]
TFTensorType = Literal["ragged", "sparse", "variable", "constant"]

class TorchRequestAttributes:
    tensorType: TorchTensorType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TorchRequestAttributesReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TorchRequestAttributesReader: ...
    @staticmethod
    def new_message() -> TorchRequestAttributesBuilder: ...
    def to_dict(self) -> dict: ...

class TorchRequestAttributesReader(TorchRequestAttributes):
    def as_builder(self) -> TorchRequestAttributesBuilder: ...

class TorchRequestAttributesBuilder(TorchRequestAttributes):
    @staticmethod
    def from_dict(dictionary: dict) -> TorchRequestAttributesBuilder: ...
    def copy(self) -> TorchRequestAttributesBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TorchRequestAttributesReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class TensorFlowRequestAttributes:
    name: str
    tensorType: TFTensorType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TensorFlowRequestAttributesReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TensorFlowRequestAttributesReader: ...
    @staticmethod
    def new_message() -> TensorFlowRequestAttributesBuilder: ...
    def to_dict(self) -> dict: ...

class TensorFlowRequestAttributesReader(TensorFlowRequestAttributes):
    def as_builder(self) -> TensorFlowRequestAttributesBuilder: ...

class TensorFlowRequestAttributesBuilder(TensorFlowRequestAttributes):
    @staticmethod
    def from_dict(dictionary: dict) -> TensorFlowRequestAttributesBuilder: ...
    def copy(self) -> TensorFlowRequestAttributesBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TensorFlowRequestAttributesReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...