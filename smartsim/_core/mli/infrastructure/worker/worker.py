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

import typing as t
from abc import ABC, abstractmethod

from .....error import SmartSimError
from .....log import get_logger
from ...comm.channel.channel import CommChannelBase
from ...infrastructure.storage.featurestore import FeatureStore, FeatureStoreKey
from ...message_handler import MessageHandler
from ...mli_schemas.model.model_capnp import Model

if t.TYPE_CHECKING:
    from smartsim._core.mli.mli_schemas.response.response_capnp import Status
    from smartsim._core.mli.mli_schemas.tensor.tensor_capnp import TensorDescriptor

logger = get_logger(__name__)


class InferenceRequest:
    """Internal representation of an inference request from a client"""

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
        """Initialize the object"""
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


class InferenceReply:
    """Internal representation of the reply to a client request for inference"""

    def __init__(
        self,
        outputs: t.Optional[t.Collection[t.Any]] = None,
        output_keys: t.Optional[t.Collection[FeatureStoreKey]] = None,
        status_enum: "Status" = "running",
        message: str = "In progress",
    ) -> None:
        """Initialize the object"""
        self.outputs: t.Collection[t.Any] = outputs or []
        self.output_keys: t.Collection[t.Optional[FeatureStoreKey]] = output_keys or []
        self.status_enum = status_enum
        self.message = message


class LoadModelResult:
    """A wrapper around a loaded model"""

    def __init__(self, model: t.Any) -> None:
        """Initialize the object"""
        self.model = model


class TransformInputResult:
    """A wrapper around a transformed input"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the object"""
        self.transformed = result


class ExecuteResult:
    """A wrapper around inference results"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the object"""
        self.predictions = result


class FetchInputResult:
    """A wrapper around fetched inputs"""

    def __init__(self, result: t.List[bytes], meta: t.Optional[t.List[t.Any]]) -> None:
        """Initialize the object"""
        self.inputs = result
        self.meta = meta


class TransformOutputResult:
    """A wrapper around inference results transformed for transmission"""

    def __init__(
        self, result: t.Any, shape: t.Optional[t.List[int]], order: str, dtype: str
    ) -> None:
        """Initialize the OutputTransformResult"""
        self.outputs = result
        self.shape = shape
        self.order = order
        self.dtype = dtype


class CreateInputBatchResult:
    """A wrapper around inputs batched into a single request"""

    def __init__(self, result: t.Any) -> None:
        """Initialize the object"""
        self.batch = result


class FetchModelResult:
    """A wrapper around raw fetched models"""

    def __init__(self, result: bytes) -> None:
        """Initialize the object"""
        self.model_bytes: bytes = result


class MachineLearningWorkerCore:
    """Basic functionality of ML worker that is shared across all worker types"""

    @staticmethod
    def deserialize_message(
        data_blob: bytes,
        channel_type: t.Type[CommChannelBase],
    ) -> InferenceRequest:
        """Deserialize a message from a byte stream into an InferenceRequest
        :param data_blob: The byte stream to deserialize
        :param channel_type: Type to be used for callback communications
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
        comm_channel = channel_type(callback_key)
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
        prepared_outputs: t.List[t.Any] = []
        if reply.output_keys:
            for value in reply.output_keys:
                if not value:
                    continue
                msg_key = MessageHandler.build_tensor_key(value.key, value.descriptor)
                prepared_outputs.append(msg_key)
        elif reply.outputs:
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
        request: InferenceRequest, feature_stores: t.Dict[str, FeatureStore]
    ) -> FetchModelResult:
        """Given a resource key, retrieve the raw model from a feature store
        :param request: The request that triggered the pipeline
        :param feature_stores: Available feature stores used for persistence
        :return: Raw bytes of the model"""

        if request.raw_model:
            # Should we cache model in the feature store?
            # model_key = hash(request.raw_model)
            # feature_store[model_key] = request.raw_model
            # short-circuit and return the directly supplied model
            return FetchModelResult(request.raw_model.data)

        if not feature_stores:
            raise ValueError("Feature store is required for model retrieval")

        if not request.model_key:
            raise SmartSimError(
                "Key must be provided to retrieve model from feature store"
            )

        key, fsd = request.model_key.key, request.model_key.descriptor

        try:
            feature_store = feature_stores[fsd]
            raw_bytes: bytes = t.cast(bytes, feature_store[key])
            return FetchModelResult(raw_bytes)
        except FileNotFoundError as ex:
            logger.exception(ex)
            raise SmartSimError(f"Model could not be retrieved with key {key}") from ex

    @staticmethod
    def fetch_inputs(
        request: InferenceRequest, feature_stores: t.Dict[str, FeatureStore]
    ) -> FetchInputResult:
        """Given a collection of ResourceKeys, identify the physical location
        and input metadata
        :param request: The request that triggered the pipeline
        :param feature_stores: Available feature stores used for persistence
        :return: the fetched input"""

        if request.raw_inputs:
            return FetchInputResult(request.raw_inputs, request.input_meta)

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
                        f"Model could not be retrieved with key {fs_key.key}"
                    ) from ex
            return FetchInputResult(
                data, meta=None
            )  # fixme: need to get both tensor and descriptor

        raise ValueError("No input source")

    @staticmethod
    def batch_requests(
        request: InferenceRequest, transform_result: TransformInputResult
    ) -> CreateInputBatchResult:
        """Create a batch of requests. Return the batch when batch_size datum have been
        collected or a configured batch duration has elapsed.
        :param request: The request that triggered the pipeline
        :param transform_result: Transformed inputs ready for batching
        :return: `None` if batch size has not been reached and timeout not exceeded."""
        if transform_result is not None or request.batch_size:
            raise NotImplementedError("Batching is not yet supported")
        return CreateInputBatchResult(None)

    @staticmethod
    def place_output(
        request: InferenceRequest,
        transform_result: TransformOutputResult,
        feature_stores: t.Dict[str, FeatureStore],
    ) -> t.Collection[t.Optional[FeatureStoreKey]]:
        """Given a collection of data, make it available as a shared resource in the
        feature store
        :param request: The request that triggered the pipeline
        :param execute_result: Results from inference
        :param feature_stores: Available feature stores used for persistence
        :return: A collection of keys that were placed in the feature store"""
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
    """Abstrct base class providing contract for a machine learning
    worker implementation."""

    @staticmethod
    @abstractmethod
    def load_model(
        request: InferenceRequest, fetch_result: FetchModelResult, device: str
    ) -> LoadModelResult:
        """Given a loaded MachineLearningModel, ensure it is loaded into
        device memory
        :param request: The request that triggered the pipeline
        :param device: The device on which the model must be placed
        :return: ModelLoadResult wrapping the model loaded for the request"""

    @staticmethod
    @abstractmethod
    def transform_input(
        request: InferenceRequest, fetch_result: FetchInputResult, device: str
    ) -> TransformInputResult:
        """Given a collection of data, perform a transformation on the data
        :param request: The request that triggered the pipeline
        :param fetch_result: Raw output from fetching inputs out of a feature store
        :param device: The device on which the transformed input must be placed
        :return: The transformed inputs wrapped in a InputTransformResult"""

    @staticmethod
    @abstractmethod
    def execute(
        request: InferenceRequest,
        load_result: LoadModelResult,
        transform_result: TransformInputResult,
    ) -> ExecuteResult:
        """Execute an ML model on inputs transformed for use by the model
        :param request: The request that triggered the pipeline
        :param load_result: The result of loading the model onto device memory
        :param transform_result: The result of transforming inputs for model consumption
        :return: The result of inference wrapped in an ExecuteResult"""

    @staticmethod
    @abstractmethod
    def transform_output(
        request: InferenceRequest, execute_result: ExecuteResult, result_device: str
    ) -> TransformOutputResult:
        """Given inference results, perform transformations required to
        transmit results to the requestor.
        :param request: The request that triggered the pipeline
        :param execute_result: The result of inference wrapped in an ExecuteResult
        :param result_device: The device on which the result of inference is placed
        :return:"""
