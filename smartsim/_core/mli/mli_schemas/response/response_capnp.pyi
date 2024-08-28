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

"""This is an automatically generated stub for `response.capnp`."""

# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal, Sequence, overload

from ..data.data_references_capnp import (
    FeatureStoreKey,
    FeatureStoreKeyBuilder,
    FeatureStoreKeyReader,
)
from ..tensor.tensor_capnp import (
    TensorDescriptor,
    TensorDescriptorBuilder,
    TensorDescriptorReader,
)
from .response_attributes.response_attributes_capnp import (
    TensorFlowResponseAttributes,
    TensorFlowResponseAttributesBuilder,
    TensorFlowResponseAttributesReader,
    TorchResponseAttributes,
    TorchResponseAttributesBuilder,
    TorchResponseAttributesReader,
)

Status = Literal["complete", "fail", "timeout", "running"]

class Response:
    class Result:
        keys: Sequence[FeatureStoreKey | FeatureStoreKeyBuilder | FeatureStoreKeyReader]
        descriptors: Sequence[
            TensorDescriptor | TensorDescriptorBuilder | TensorDescriptorReader
        ]
        def which(self) -> Literal["keys", "descriptors"]: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Response.ResultReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Response.ResultReader: ...
        @staticmethod
        def new_message() -> Response.ResultBuilder: ...
        def to_dict(self) -> dict: ...

    class ResultReader(Response.Result):
        keys: Sequence[FeatureStoreKeyReader]
        descriptors: Sequence[TensorDescriptorReader]
        def as_builder(self) -> Response.ResultBuilder: ...

    class ResultBuilder(Response.Result):
        keys: Sequence[FeatureStoreKey | FeatureStoreKeyBuilder | FeatureStoreKeyReader]
        descriptors: Sequence[
            TensorDescriptor | TensorDescriptorBuilder | TensorDescriptorReader
        ]
        @staticmethod
        def from_dict(dictionary: dict) -> Response.ResultBuilder: ...
        def copy(self) -> Response.ResultBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Response.ResultReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    class CustomAttributes:
        torch: (
            TorchResponseAttributes
            | TorchResponseAttributesBuilder
            | TorchResponseAttributesReader
        )
        tf: (
            TensorFlowResponseAttributes
            | TensorFlowResponseAttributesBuilder
            | TensorFlowResponseAttributesReader
        )
        none: None
        def which(self) -> Literal["torch", "tf", "none"]: ...
        @overload
        def init(self, name: Literal["torch"]) -> TorchResponseAttributes: ...
        @overload
        def init(self, name: Literal["tf"]) -> TensorFlowResponseAttributes: ...
        @staticmethod
        @contextmanager
        def from_bytes(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Iterator[Response.CustomAttributesReader]: ...
        @staticmethod
        def from_bytes_packed(
            data: bytes,
            traversal_limit_in_words: int | None = ...,
            nesting_limit: int | None = ...,
        ) -> Response.CustomAttributesReader: ...
        @staticmethod
        def new_message() -> Response.CustomAttributesBuilder: ...
        def to_dict(self) -> dict: ...

    class CustomAttributesReader(Response.CustomAttributes):
        torch: TorchResponseAttributesReader
        tf: TensorFlowResponseAttributesReader
        def as_builder(self) -> Response.CustomAttributesBuilder: ...

    class CustomAttributesBuilder(Response.CustomAttributes):
        torch: (
            TorchResponseAttributes
            | TorchResponseAttributesBuilder
            | TorchResponseAttributesReader
        )
        tf: (
            TensorFlowResponseAttributes
            | TensorFlowResponseAttributesBuilder
            | TensorFlowResponseAttributesReader
        )
        @staticmethod
        def from_dict(dictionary: dict) -> Response.CustomAttributesBuilder: ...
        def copy(self) -> Response.CustomAttributesBuilder: ...
        def to_bytes(self) -> bytes: ...
        def to_bytes_packed(self) -> bytes: ...
        def to_segments(self) -> list[bytes]: ...
        def as_reader(self) -> Response.CustomAttributesReader: ...
        @staticmethod
        def write(file: BufferedWriter) -> None: ...
        @staticmethod
        def write_packed(file: BufferedWriter) -> None: ...

    status: Status
    message: str
    result: Response.Result | Response.ResultBuilder | Response.ResultReader
    customAttributes: (
        Response.CustomAttributes
        | Response.CustomAttributesBuilder
        | Response.CustomAttributesReader
    )
    @overload
    def init(self, name: Literal["result"]) -> Result: ...
    @overload
    def init(self, name: Literal["customAttributes"]) -> CustomAttributes: ...
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[ResponseReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> ResponseReader: ...
    @staticmethod
    def new_message() -> ResponseBuilder: ...
    def to_dict(self) -> dict: ...

class ResponseReader(Response):
    result: Response.ResultReader
    customAttributes: Response.CustomAttributesReader
    def as_builder(self) -> ResponseBuilder: ...

class ResponseBuilder(Response):
    result: Response.Result | Response.ResultBuilder | Response.ResultReader
    customAttributes: (
        Response.CustomAttributes
        | Response.CustomAttributesBuilder
        | Response.CustomAttributesReader
    )
    @staticmethod
    def from_dict(dictionary: dict) -> ResponseBuilder: ...
    def copy(self) -> ResponseBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> ResponseReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
