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
from ...comm.channel.channel import CommChannelBase
from ...message_handler import MessageHandler
from ...mli_schemas.model.model_capnp import Model
from ..storage.feature_store import FeatureStore, FeatureStoreKey

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.data.data_references_capnp import TensorKey
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status
    from smartsim._core.mli.mli_schemas.tensor.tensor_capnp import TensorDescriptor

logger = get_logger(__name__)

# Placeholder
ModelIdentifier = FeatureStoreKey


class InferenceRequest:
    """Internal representation of an inference request from a client."""

    def __init__(
        self,
        model_key: t.Optional[FeatureStoreKey] = None,
        callback: t.Optional[CommChannelBase] = None,
        raw_inputs: t.Optional[t.List[bytes]] = None,
        input_keys: t.Optional[t.List[FeatureStoreKey]] = None,
        input_meta: t.Optional[t.List[t.Any]] = None,
        output_keys: t.Optional[t.List[FeatureStoreKey]] = None,
        raw_model: t.Optional[Model] = None,
        batch_size: int = 0,
    ):
        """Initialize the InferenceRequest.

        :param model_key: A tuple containing a (key, descriptor) pair
        :param callback: The channel used for notification of inference completion
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
        self.callback = callback
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
    def has_raw_model(self):
        return self.raw_model is not None
    
    @property
    def has_model_key(self):
        return self.model_key is not None


class InferenceReply:
    """Internal representation of the reply to a client request for inference."""

    def __init__(
        self,
        outputs: t.Optional[t.Collection[t.Any]] = None,
        output_keys: t.Optional[t.Collection[FeatureStoreKey]] = None,
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
        self.output_keys: t.Collection[t.Optional[FeatureStoreKey]] = output_keys or []
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


class LoadModelResult:
    """A wrapper around a loaded model."""

    def __init__(self, model: t.Any) -> None:
        """Initialize the LoadModelResult.

        :param model: The loaded model
        """
        self.model = model


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

    def __init__(self, result: t.List[bytes], meta: t.Optional[t.List[t.Any]]) -> None:
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

    requests: list[InferenceRequest]
    """List of InferenceRequests in the batch"""
    inputs: t.Optional[TransformInputResult]
    """Transformed batch of input tensors"""
    model_id: ModelIdentifier
    """Model (key, descriptor) tuple"""

    @property
    def has_valid_requests(self) -> bool:
        """Returns whether the batch contains at least one request.

        :returns: True if at least one request is available
        """
        return len(self.requests) > 0

    @property
    def has_inputs(self) -> bool:
        """Returns whether the batch has inputs.

        :returns: True if the batch has inputs
        """
        return self.inputs is not None and bool(self.inputs)

    @property
    def has_raw_model(self) -> bool:
        """Returns whether the batch has a raw model.

        :returns: True if the batch has a raw model
        """
        return self.raw_model is not None

    @property
    def raw_model(self) -> t.Optional[t.Any]:
        """Returns the raw model to use to execute for this batch
        if it is available.

        :returns: A model if available, otherwise None"""
        if self.has_valid_requests:
            return self.requests[0].raw_model
        return None

    @property
    def input_keys(self) -> t.List[FeatureStoreKey]:
        """All input keys available in this batch's requests.

        :returns: All input keys belonging to requests in this batch"""
        keys = []
        for request in self.requests:
            keys.extend(request.input_keys)

        return keys

    @property
    def output_keys(self) -> t.List[FeatureStoreKey]:
        """All output keys available in this batch's requests.

        :returns: All output keys belonging to requests in this batch"""
        keys = []
        for request in self.requests:
            keys.extend(request.output_keys)

        return keys


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types."""

    @staticmethod
    def deserialize_message(
        data_blob: bytes,
        callback_factory: t.Callable[[bytes], CommChannelBase],
    ) -> InferenceRequest:
        """Deserialize a message from a byte stream into an InferenceRequest.

        :param data_blob: The byte stream to deserialize
        :param callback_factory: A factory method that can create an instance
        of the desired concrete comm channel type
        :returns: The raw input message deserialized into an InferenceRequest
        """
        request = MessageHandler.deserialize_request(data_blob)
        model_key: t.Optional[FeatureStoreKey] = None
        model_bytes: t.Optional[Model] = None

        if request.model.which() == "key":
            model_key = FeatureStoreKey(
                key=request.model.key.key,
                descriptor=request.model.key.featureStoreDescriptor,
            )
        elif request.model.which() == "data":
            model_bytes = request.model.data

        callback_key = request.replyChannel.descriptor
        comm_channel = callback_factory(callback_key)
        input_keys: t.Optional[t.List[FeatureStoreKey]] = None
        input_bytes: t.Optional[t.List[bytes]] = None
        output_keys: t.Optional[t.List[FeatureStoreKey]] = None
        input_meta: t.Optional[t.List[TensorDescriptor]] = None

        if request.input.which() == "keys":
            input_keys = [
                FeatureStoreKey(key=value.key, descriptor=value.featureStoreDescriptor)
                for value in request.input.keys
            ]
        elif request.input.which() == "descriptors":
            input_meta = request.input.descriptors  # type: ignore

        if request.output:
            output_keys = [
                FeatureStoreKey(key=value.key, descriptor=value.featureStoreDescriptor)
                for value in request.output
            ]

        inference_request = InferenceRequest(
            model_key=model_key,
            callback=comm_channel,
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
        model is not provided"""

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
        except FileNotFoundError as ex:
            logger.exception(ex)
            raise SmartSimError(f"Model could not be retrieved with key {key}") from ex

    @staticmethod
    def fetch_inputs(
        batch: RequestBatch, feature_stores: t.Dict[str, FeatureStore]
    ) -> t.List[FetchInputResult]:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata.

        :param batch: The batch of requests that triggered the pipeline
        :param feature_stores: Available feature stores used for persistence
        :returns: The fetched input
        :raises ValueError: If neither an input key or an input tensor are provided
        :raises SmartSimError: If a tensor for a given key cannot be retrieved"""
        fetch_results = []
        for request in batch.requests:
            if request.raw_inputs:
                fetch_results.append(
                    FetchInputResult(request.raw_inputs, request.input_meta)
                )
                continue

            if not feature_stores:
                raise ValueError("No input and no feature store provided")

            if request.input_keys:
                data: t.List[bytes] = []

                for fs_key in request.input_keys:
                    try:
                        feature_store = feature_stores[fs_key.descriptor]
                        tensor_bytes = t.cast(bytes, feature_store[fs_key.key])
                        data.append(tensor_bytes)
                    except KeyError as ex:
                        logger.exception(ex)
                        raise SmartSimError(
                            f"Tensor could not be retrieved with key {fs_key.key}"
                        ) from ex
                fetch_results.append(
                    FetchInputResult(data, meta=None)
                )  # fixme: need to get both tensor and descriptor
                continue

            raise ValueError("No input source")

        return fetch_results

    @staticmethod
    def place_output(
        request: InferenceRequest,
        transform_result: TransformOutputResult,
        feature_stores: t.Dict[str, FeatureStore],
    ) -> t.Collection[t.Optional[FeatureStoreKey]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store.

        :param request: The request that triggered the pipeline
        :param execute_result: Results from inference
        :param feature_stores: Available feature stores used for persistence
        :returns: A collection of keys that were placed in the feature store
        :raises ValueError: If a feature store is not provided
        """
        if not feature_stores:
            raise ValueError("Feature store is required for output persistence")

        keys: t.List[t.Optional[FeatureStoreKey]] = []
        # need to decide how to get back to original sub-batch inputs so they can be
        # accurately placed, datum might need to include this.

        # Consider parallelizing all PUT feature_store operations
        for fs_key, v in zip(request.output_keys, transform_result.outputs):
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
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory.

        :param request: The request that triggered the pipeline
        :param device: The device on which the model must be placed
        :returns: LoadModelResult wrapping the model loaded for the request"""

    @staticmethod
    @abstractmethod
    def transform_input(
        batch: RequestBatch,
        fetch_results: list[FetchInputResult],
        mem_pool: MemoryPool,
    ) -> TransformInputResult:
        """Given a collection of data, perform a transformation on the data and put
        the raw tensor data on a MemoryPool allocation.

        :param request: The request that triggered the pipeline
        :param fetch_result: Raw outputs from fetching inputs out of a feature store
        :param mem_pool: The memory pool used to access batched input tensors
        :returns: The transformed inputs wrapped in a TransformInputResult"""

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
        :returns: The result of inference wrapped in an ExecuteResult"""

    @staticmethod
    @abstractmethod
    def transform_output(
        batch: RequestBatch, execute_result: ExecuteResult
    ) -> t.List[TransformOutputResult]:
        """Given inference results, perform transformations required to
        transmit results to the requestor.

        :param batch: The batch of requests that triggered the pipeline
        :param execute_result: The result of inference wrapped in an ExecuteResult
        :returns: A list of transformed outputs"""
