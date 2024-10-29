# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
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

# pylint: disable=import-error
from dragon.managed_memory import MemoryPool

# isort: off
# isort: on

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .....error import SmartSimError
from .....log import get_logger
from ...message_handler import MessageHandler
from ...mli_schemas.model.model_capnp import Model
from ...mli_schemas.tensor.tensor_capnp import TensorDescriptor
from ..storage.feature_store import FeatureStore, ModelKey, TensorKey

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status

logger = get_logger(__name__)

# Placeholder
ModelIdentifier = ModelKey


class InferenceRequest:
    """Internal representation of an inference request from a client."""

    def __init__(
        self,
        model_key: t.Optional[ModelKey] = None,
        callback_desc: t.Optional[str] = None,
        raw_inputs: t.Optional[t.List[bytes]] = None,
        input_keys: t.Optional[t.List[TensorKey]] = None,
        input_meta: t.Optional[t.List[TensorDescriptor]] = None,
        output_keys: t.Optional[t.List[TensorKey]] = None,
        raw_model: t.Optional[Model] = None,
        batch_size: int = 0,
    ):
        """Initialize the InferenceRequest.

        :param model_key: A tuple containing a (key, descriptor) pair
        :param callback_desc: The channel descriptor used for notification
        of inference completion
        :param raw_inputs: Raw bytes of tensor inputs
        :param input_keys: A list of tuples containing a (key, descriptor) pair
        :param input_meta: Metadata about the input data
        :param output_keys: A list of tuples containing a (key, descriptor) pair
        :param raw_model: Raw bytes of an ML model
        :param batch_size: The batch size to apply when batching
        """
        self.model_key = model_key
        """A tuple containing a (key, descriptor) pair"""
        self.raw_model = raw_model
        """Raw bytes of an ML model"""
        self.callback_desc = callback_desc
        """The channel used for notification of inference completion"""
        self.raw_inputs = raw_inputs or []
        """Raw bytes of tensor inputs"""
        self.input_keys = input_keys or []
        """A list of tuples containing a (key, descriptor) pair"""
        self.input_meta = input_meta or []
        """Metadata about the input data"""
        self.output_keys = output_keys or []
        """A list of tuples containing a (key, descriptor) pair"""
        self.batch_size = batch_size
        """The batch size to apply when batching"""

    @property
    def has_raw_model(self) -> bool:
        """Check if the InferenceRequest contains a raw_model.

        :returns: True if raw_model is not None, False otherwise
        """
        return self.raw_model is not None

    @property
    def has_model_key(self) -> bool:
        """Check if the InferenceRequest contains a model_key.

        :returns: True if model_key is not None, False otherwise
        """
        return self.model_key is not None

    @property
    def has_raw_inputs(self) -> bool:
        """Check if the InferenceRequest contains raw_inputs.

        :returns: True if raw_outputs is not None and is not an empty list,
        False otherwise
        """
        return self.raw_inputs is not None and bool(self.raw_inputs)

    @property
    def has_input_keys(self) -> bool:
        """Check if the InferenceRequest contains input_keys.

        :returns: True if input_keys is not None and is not an empty list,
        False otherwise
        """
        return self.input_keys is not None and bool(self.input_keys)

    @property
    def has_output_keys(self) -> bool:
        """Check if the InferenceRequest contains output_keys.

        :returns: True if output_keys is not None and is not an empty list,
        False otherwise
        """
        return self.output_keys is not None and bool(self.output_keys)

    @property
    def has_input_meta(self) -> bool:
        """Check if the InferenceRequest contains input_meta.

        :returns: True if input_meta is not None and is not an empty list,
        False otherwise
        """
        return self.input_meta is not None and bool(self.input_meta)


class InferenceReply:
    """Internal representation of the reply to a client request for inference."""

    def __init__(
        self,
        outputs: t.Optional[t.Collection[t.Any]] = None,
        output_keys: t.Optional[t.Collection[TensorKey]] = None,
        status_enum: "Status" = "running",
        message: str = "In progress",
    ) -> None:
        """Initialize the InferenceReply.

        :param outputs: List of output data
        :param output_keys: List of keys used for output data
        :param status_enum: Status of the reply
        :param message: Status message that corresponds with the status enum
        """
        self.outputs: t.Collection[t.Any] = outputs or []
        """List of output data"""
        self.output_keys: t.Collection[t.Optional[TensorKey]] = output_keys or []
        """List of keys used for output data"""
        self.status_enum = status_enum
        """Status of the reply"""
        self.message = message
        """Status message that corresponds with the status enum"""

    @property
    def has_outputs(self) -> bool:
        """Check if the InferenceReply contains outputs.

        :returns: True if outputs is not None and is not an empty list,
        False otherwise
        """
        return self.outputs is not None and bool(self.outputs)

    @property
    def has_output_keys(self) -> bool:
        """Check if the InferenceReply contains output_keys.

        :returns: True if output_keys is not None and is not an empty list,
        False otherwise
        """
        return self.output_keys is not None and bool(self.output_keys)


@dataclass
class TensorMeta:
    """Metadata about a tensor, built from TensorDescriptors."""

    dimensions: t.List[int]
    """Dimensions of the tensor"""
    order: str
    """Order of the tensor in row major ("c"), or
    column major ("f") format"""
    datatype: str
    """Datatype of the tensor as specified by the TensorDescriptor
    NumericalType enums. Examples include "float32", "int8", etc."""


class LoadModelResult:
    """A wrapper around a loaded model."""

    def __init__(self, model: t.Any) -> None:
        """Initialize the LoadModelResult.

        :param model: The loaded model
        """
        self.model = model
        """The loaded model (e.g. a TensorFlow, PyTorch, ONNX, etc. model)"""


class TransformInputResult:
    """A wrapper around a transformed batch of input tensors"""

    def __init__(
        self,
        result: t.Any,
        slices: list[slice],
        dims: list[list[int]],
        dtypes: list[str],
    ) -> None:
        """Initialize the TransformInputResult.

        :param result: List of Dragon MemoryAlloc objects on which
        the tensors are stored
        :param slices: The slices that represent which portion of the
        input tensors belongs to which request
        :param dims: Dimension of the transformed tensors
        :param dtypes: Data type of transformed tensors
        """
        self.transformed = result
        """List of Dragon MemoryAlloc objects on which the tensors are stored"""
        self.slices = slices
        """Each slice represents which portion of the input tensors belongs to
        which request"""
        self.dims = dims
        """Dimension of the transformed tensors"""
        self.dtypes = dtypes
        """Data type of transformed tensors"""


class ExecuteResult:
    """A wrapper around inference results."""

    def __init__(self, result: t.Any, slices: list[slice]) -> None:
        """Initialize the ExecuteResult.

        :param result: Result of the execution
        :param slices: The slices that represent which portion of the input
        tensors belongs to which request
        """
        self.predictions = result
        """Result of the execution"""
        self.slices = slices
        """The slices that represent which portion of the input
        tensors belongs to which request"""


class FetchInputResult:
    """A wrapper around fetched inputs."""

    def __init__(
        self,
        result: t.List[t.List[bytes]],
        meta: t.List[t.List[t.Optional[TensorMeta]]],
    ) -> None:
        """Initialize the FetchInputResult.

        :param result: List of input tensor bytes
        :param meta: List of metadata that corresponds with the inputs
        """
        self.inputs = result
        """List of input tensor bytes"""
        self.meta = meta
        """List of metadata that corresponds with the inputs"""


class TransformOutputResult:
    """A wrapper around inference results transformed for transmission."""

    def __init__(
        self, result: t.Any, shape: t.Optional[t.List[int]], order: str, dtype: str
    ) -> None:
        """Initialize the TransformOutputResult.

        :param result: Transformed output results
        :param shape: Shape of output results
        :param order: Order of output results
        :param dtype: Datatype of output results
        """
        self.outputs = result
        """Transformed output results"""
        self.shape = shape
        """Shape of output results"""
        self.order = order
        """Order of output results"""
        self.dtype = dtype
        """Datatype of output results"""


class CreateInputBatchResult:
    """A wrapper around inputs batched into a single request."""

    def __init__(self, result: t.Any) -> None:
        """Initialize the CreateInputBatchResult.

        :param result: Inputs batched into a single request
        """
        self.batch = result
        """Inputs batched into a single request"""


class FetchModelResult:
    """A wrapper around raw fetched models."""

    def __init__(self, result: bytes) -> None:
        """Initialize the FetchModelResult.

        :param result: The raw fetched model
        """
        self.model_bytes: bytes = result
        """The raw fetched model"""


@dataclass
class RequestBatch:
    """A batch of aggregated inference requests."""

    raw_model: t.Optional[Model]
    """Raw bytes of the model"""
    callback_descriptors: t.List[str]
    """The descriptors for channels used for notification of inference completion"""
    raw_inputs: t.List[t.List[bytes]]
    """Raw bytes of tensor inputs"""
    input_meta: t.List[t.List[TensorMeta]]
    """Metadata about the input data"""
    input_keys: t.List[t.List[TensorKey]]
    """A list of tuples containing a (key, descriptor) pair"""
    output_key_refs: t.Dict[str, t.List[TensorKey]]
    """A dictionary mapping callbacks descriptors to output keys"""
    inputs: t.Optional[TransformInputResult]
    """Transformed batch of input tensors"""
    model_id: "ModelIdentifier"
    """Model (key, descriptor) tuple"""

    @property
    def has_callbacks(self) -> bool:
        """Determines if the batch has at least one callback channel
        available for sending results.

        :returns: True if at least one callback is present
        """
        return len(self.callback_descriptors) > 0

    @property
    def has_raw_model(self) -> bool:
        """Returns whether the batch has a raw model.

        :returns: True if the batch has a raw model
        """
        return self.raw_model is not None

    @classmethod
    def from_requests(
        cls,
        requests: t.List[InferenceRequest],
        model_id: ModelIdentifier,
    ) -> "RequestBatch":
        """Create a RequestBatch from a list of requests.

        :param requests: The requests to batch
        :param model_id: The model identifier
        :returns: A RequestBatch instance
        """
        return cls(
            raw_model=requests[0].raw_model,
            callback_descriptors=[
                request.callback_desc for request in requests if request.callback_desc
            ],
            raw_inputs=[
                request.raw_inputs for request in requests if request.raw_inputs
            ],
            input_meta=[
                [
                    TensorMeta(
                        dimensions=list(meta.dimensions),
                        order=str(meta.order),
                        datatype=str(meta.dataType),
                    )
                    for meta in request.input_meta
                ]
                for request in requests
                if request.input_meta
            ],
            input_keys=[
                request.input_keys for request in requests if request.input_keys
            ],
            output_key_refs={
                request.callback_desc: request.output_keys
                for request in requests
                if request.callback_desc and request.output_keys
            },
            inputs=None,
            model_id=model_id,
        )


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types."""

    @staticmethod
    def deserialize_message(
        data_blob: bytes,
    ) -> InferenceRequest:
        """Deserialize a message from a byte stream into an InferenceRequest.

        :param data_blob: The byte stream to deserialize
        :returns: The raw input message deserialized into an InferenceRequest
        """
        request = MessageHandler.deserialize_request(data_blob)
        model_key: t.Optional[ModelKey] = None
        model_bytes: t.Optional[Model] = None

        if request.model.which() == "key":
            model_key = ModelKey(
                key=request.model.key.key,
                descriptor=request.model.key.descriptor,
            )
        elif request.model.which() == "data":
            model_bytes = request.model.data

        callback_key = request.replyChannel.descriptor
        input_keys: t.Optional[t.List[TensorKey]] = None
        input_bytes: t.Optional[t.List[bytes]] = None
        output_keys: t.Optional[t.List[TensorKey]] = None
        input_meta: t.Optional[t.List[TensorDescriptor]] = None

        if request.input.which() == "keys":
            input_keys = [
                TensorKey(key=value.key, descriptor=value.descriptor)
                for value in request.input.keys
            ]
        elif request.input.which() == "descriptors":
            input_meta = request.input.descriptors  # type: ignore

        if request.output:
            output_keys = [
                TensorKey(key=value.key, descriptor=value.descriptor)
                for value in request.output
            ]

        inference_request = InferenceRequest(
            model_key=model_key,
            callback_desc=callback_key,
            raw_inputs=input_bytes,
            input_meta=input_meta,
            input_keys=input_keys,
            output_keys=output_keys,
            raw_model=model_bytes,
            batch_size=0,
        )
        return inference_request

    @staticmethod
    def prepare_outputs(reply: InferenceReply) -> t.List[t.Any]:
        """Assemble the output information based on whether the output
        information will be in the form of TensorKeys or TensorDescriptors.

        :param reply: The reply that the output belongs to
        :returns: The list of prepared outputs, depending on the output
        information needed in the reply
        """
        prepared_outputs: t.List[t.Any] = []
        if reply.has_output_keys:
            for value in reply.output_keys:
                if not value:
                    continue
                msg_key = MessageHandler.build_tensor_key(value.key, value.descriptor)
                prepared_outputs.append(msg_key)
        elif reply.has_outputs:
            for _ in reply.outputs:
                msg_tensor_desc = MessageHandler.build_tensor_descriptor(
                    "c",
                    "float32",
                    [1],
                )
                prepared_outputs.append(msg_tensor_desc)
        return prepared_outputs

    @staticmethod
    def fetch_model(
        batch: RequestBatch, feature_stores: t.Dict[str, FeatureStore]
    ) -> FetchModelResult:
        """Given a resource key, retrieve the raw model from a feature store.

        :param batch: The batch of requests that triggered the pipeline
        :param feature_stores: Available feature stores used for persistence
        :returns: Raw bytes of the model
        :raises SmartSimError: If neither a key or a model are provided or the
        model cannot be retrieved from the feature store
        :raises ValueError: If a feature store is not available and a raw
        model is not provided
        """
        # All requests in the same batch share the model
        if batch.raw_model:
            return FetchModelResult(batch.raw_model.data)

        if not feature_stores:
            raise ValueError("Feature store is required for model retrieval")

        if batch.model_id is None:
            raise SmartSimError(
                "Key must be provided to retrieve model from feature store"
            )

        key, fsd = batch.model_id.key, batch.model_id.descriptor

        try:
            feature_store = feature_stores[fsd]
            raw_bytes: bytes = t.cast(bytes, feature_store[key])
            return FetchModelResult(raw_bytes)
        except (FileNotFoundError, KeyError) as ex:
            logger.exception(ex)
            raise SmartSimError(f"Model could not be retrieved with key {key}") from ex

    @staticmethod
    def fetch_inputs(
        batch: RequestBatch, feature_stores: t.Dict[str, FeatureStore]
    ) -> FetchInputResult:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata.

        :param batch: The batch of requests that triggered the pipeline
        :param feature_stores: Available feature stores used for persistence
        :returns: The fetched input
        :raises ValueError: If neither an input key or an input tensor are provided
        :raises SmartSimError: If a tensor for a given key cannot be retrieved
        """
        if not batch.raw_inputs and not batch.input_keys:
            raise ValueError("No input source")

        if not feature_stores:
            raise ValueError("No feature stores provided")

        data_list: t.List[t.List[bytes]] = []
        meta_list: t.List[t.List[t.Optional[TensorMeta]]] = []
        # meta_list will be t.List[t.List[TensorMeta]] once input_key metadata
        # is available to be retrieved from the feature store

        if batch.raw_inputs:
            for raw_inputs, input_meta in zip(batch.raw_inputs, batch.input_meta):
                data_list.append(raw_inputs)
                meta_list.append(input_meta)  # type: ignore

        if batch.input_keys:
            for batch_keys in batch.input_keys:
                batch_data: t.List[bytes] = []
                for fs_key in batch_keys:
                    try:
                        feature_store = feature_stores[fs_key.descriptor]
                        tensor_bytes = t.cast(bytes, feature_store[fs_key.key])
                        batch_data.append(tensor_bytes)
                    except KeyError as ex:
                        logger.exception(ex)
                        raise SmartSimError(
                            f"Tensor could not be retrieved with key {fs_key.key}"
                        ) from ex
                data_list.append(batch_data)
                meta_list.append([None] * len(batch_data))
                # fixme: need to get both tensor and descriptor
                # this will eventually append meta info retrieved from the feature store

        return FetchInputResult(result=data_list, meta=meta_list)

    @staticmethod
    def place_output(
        output_keys: t.List[TensorKey],
        transform_result: TransformOutputResult,
        feature_stores: t.Dict[str, FeatureStore],
    ) -> t.Collection[t.Optional[TensorKey]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store.

        :param output_keys: The output_keys that will be placed in the feature store
        :param transform_result: Transformed version of the inference result
        :param feature_stores: Available feature stores used for persistence
        :returns: A collection of keys that were placed in the feature store
        :raises ValueError: If a feature store is not provided
        """
        if not feature_stores:
            raise ValueError("Feature store is required for output persistence")

        keys: t.List[t.Optional[TensorKey]] = []
        # need to decide how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.

        # Consider parallelizing all PUT feature_store operations
        for fs_key, v in zip(output_keys, transform_result.outputs):
            feature_store = feature_stores[fs_key.descriptor]
            feature_store[fs_key.key] = v
            keys.append(fs_key)

        return keys


class MachineLearningWorkerBase(MachineLearningWorkerCore, ABC):
    """Abstract base class providing contract for a machine learning
    worker implementation."""

    @staticmethod
    @abstractmethod
    def load_model(
        batch: RequestBatch, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        """Given the raw bytes of an ML model that were fetched, ensure
        it is loaded into device memory.

        :param request: The request that triggered the pipeline
        :param fetch_result: The result of a fetch-model operation; contains
        the raw bytes of the ML model.
        :param device: The device on which the model must be placed
        :returns: LoadModelResult wrapping the model loaded for the request
        :raises ValueError: If model reference object is not found
        :raises RuntimeError: If loading and evaluating the model failed
        """

    @staticmethod
    @abstractmethod
    def transform_input(
        batch: RequestBatch,
        fetch_results: FetchInputResult,
        mem_pool: MemoryPool,
    ) -> TransformInputResult:
        """Given a collection of data, perform a transformation on the data and put
        the raw tensor data on a MemoryPool allocation.

        :param batch: The request that triggered the pipeline
        :param fetch_result: Raw outputs from fetching inputs out of a feature store
        :param mem_pool: The memory pool used to access batched input tensors
        :returns: The transformed inputs wrapped in a TransformInputResult
        :raises ValueError: If tensors cannot be reconstructed
        :raises IndexError: If index out of range
        """

    @staticmethod
    @abstractmethod
    def execute(
        batch: RequestBatch,
        load_result: LoadModelResult,
        transform_result: TransformInputResult,
        device: str,
    ) -> ExecuteResult:
        """Execute an ML model on inputs transformed for use by the model.

        :param batch: The batch of requests that triggered the pipeline
        :param load_result: The result of loading the model onto device memory
        :param transform_result: The result of transforming inputs for model consumption
        :param device: The device on which the model will be executed
        :returns: The result of inference wrapped in an ExecuteResult
        :raises SmartSimError: If model is not loaded
        :raises IndexError: If memory slicing is out of range
        :raises ValueError: If tensor creation fails or is unable to evaluate the model
        """

    @staticmethod
    @abstractmethod
    def transform_output(
        batch: RequestBatch, execute_result: ExecuteResult
    ) -> t.List[TransformOutputResult]:
        """Given inference results, perform transformations required to
        transmit results to the requestor.

        :param batch: The batch of requests that triggered the pipeline
        :param execute_result: The result of inference wrapped in an ExecuteResult
        :returns: A list of transformed outputs
        :raises IndexError: If indexing is out of range
        :raises ValueError: If transforming output fails
        """
