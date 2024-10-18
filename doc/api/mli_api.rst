************************************
Machine Learning Infrastructure API
************************************


DragonCommChannel
=================

.. currentmodule:: smartsim._core.mli.comm.channel.dragon_channel


.. autosummary::

   DragonCommChannel.channel
   DragonCommChannel.send
   DragonCommChannel.recv
   DragonCommChannel.from_descriptor
   DragonCommChannel.from_local


.. autoclass:: DragonCommChannel
   :show-inheritance:
   :members:


DragonFLIChannel
================

.. currentmodule:: smartsim._core.mli.comm.channel.dragon_fli


.. autosummary::

   DragonFLIChannel.__init__
   DragonFLIChannel.send
   DragonFLIChannel.send_multiple
   DragonFLIChannel.recv
   DragonFLIChannel.from_descriptor

.. autoclass:: DragonFLIChannel
   :show-inheritance:
   :members:


EventBroadcaster
================

.. currentmodule:: smartsim._core.mli.infrastructure.comm.broadcaster


.. autosummary::

   EventBroadcaster.__init__
   EventBroadcaster.name
   EventBroadcaster.num_buffered
   EventBroadcaster.send

.. autoclass:: EventBroadcaster
   :show-inheritance:
   :members:


EventConsumer
=============

.. currentmodule:: smartsim._core.mli.infrastructure.comm.consumer

.. autosummary::

   EventConsumer.__init__
   EventConsumer.descriptor
   EventConsumer.name
   EventConsumer.recv
   EventConsumer.register 
   EventConsumer.unregister
   EventConsumer.listen_once
   EventConsumer.listen


.. autoclass:: EventConsumer
   :show-inheritance:
   :members:


.. currentmodule:: smartsim._core.mli.infrastructure.comm.event

EventBase
==========

.. autosummary::
   EventBase.__init__

.. autoclass:: EventBase
   :show-inheritance:
   :members:

OnShutdownRequested
===================

.. autosummary::
   
   OnShutdownRequested.__init__

.. autoclass:: OnShutdownRequested
   :show-inheritance:
   :members:


OnCreateConsumer
================

.. autosummary::
   
   OnCreateConsumer.__init__

.. autoclass:: OnCreateConsumer
   :show-inheritance:
   :members:


OnRemoveConsumer
================

.. autosummary::
   
   OnRemoveConsumer.__init__

.. autoclass:: OnRemoveConsumer
   :show-inheritance:
   :members:


OnWriteFeatureStore
===================

.. autosummary::
   
   OnWriteFeatureStore.__init__

.. autoclass:: OnWriteFeatureStore
   :show-inheritance:
   :members:


EventProducer
==============

.. currentmodule:: smartsim._core.mli.infrastructure.comm.producer

.. autosummary::

   EventProducer.send

.. autoclass:: EventProducer
   :show-inheritance:
   :members:


.. currentmodule:: smartsim._core.mli.infrastructure.control.device_manager

WorkerDevice
=============

.. autosummary::

   WorkerDevice.name
   WorkerDevice.add_model
   WorkerDevice.remove_model
   WorkerDevice.get_model
   WorkerDevice.get

.. autoclass:: WorkerDevice
   :show-inheritance:
   :members:


DeviceManager
==============

.. autosummary::

   DeviceManager.get_device

.. autoclass:: DeviceManager
   :show-inheritance:
   :members:


ConsumerRegistrationListener
============================

.. currentmodule:: smartsim._core.mli.infrastructure.control.listener 


.. autosummary::

   ConsumerRegistrationListener.__init__

.. autoclass:: ConsumerRegistrationListener
   :show-inheritance:
   :members:


.. currentmodule:: smartsim._core.mli.infrastructure.control.request_dispatcher

BatchQueue
==========

.. autosummary::

   BatchQueue.__init__
   BatchQueue.uid 
   BatchQueue.model_id
   BatchQueue.put 
   BatchQueue.ready 
   BatchQueue.make_disposable
   BatchQueue.can_be_removed
   BatchQueue.flush
   BatchQueue.full
   BatchQueue.empty

.. autoclass:: BatchQueue
   :show-inheritance:
   :members:


RequestDispatcher
=================

.. autosummary::

   RequestDispatcher.__init__
   RequestDispatcher.has_featurestore_factory
   RequestDispatcher.remove_queues
   RequestDispatcher.task_queue
   RequestDispatcher.dispatch
   RequestDispatcher.flush_requests

.. autoclass:: RequestDispatcher
   :show-inheritance:
   :members:



WorkerManager
=============

.. currentmodule:: smartsim._core.mli.infrastructure.control.worker_manager

.. autosummary::

   WorkerManager.__init__
   WorkerManager.has_featurestore_factory

.. autoclass:: WorkerManager
   :show-inheritance:
   :members:


   
BackboneFeatureStore
====================

.. currentmodule:: smartsim._core.mli.infrastructure.storage.backbone_feature_store

.. autosummary::

   BackboneFeatureStore.__init__
   BackboneFeatureStore.wait_timeout
   BackboneFeatureStore.notification_channels
   BackboneFeatureStore.backend_channel
   BackboneFeatureStore.worker_queue
   BackboneFeatureStore.creation_date
   BackboneFeatureStore.from_writable_descriptor
   BackboneFeatureStore.wait_for
   BackboneFeatureStore.get_env

.. autoclass:: BackboneFeatureStore
   :show-inheritance:
   :members:


DragonFeatureStore
==================

.. currentmodule:: smartsim._core.mli.infrastructure.storage.dragon_feature_store

.. autosummary::

   DragonFeatureStore.__init__
   DragonFeatureStore.pop 
   DragonFeatureStore.from_descriptor

.. autoclass:: DragonFeatureStore
   :show-inheritance:
   :members:


.. currentmodule:: smartsim._core.mli.infrastructure.storage.feature_store

ReservedKeys
============

.. autosummary::

   ReservedKeys.contains

.. autoclass:: ReservedKeys
   :show-inheritance:
   :members:


TensorKey
=========

.. autoclass:: TensorKey
   :show-inheritance:
   :members:


ModelKey
========

.. autoclass:: ModelKey
   :show-inheritance:
   :members:


FeatureStore
============

.. autosummary::

   FeatureStore.__init__
   FeatureStore.descriptor

.. autoclass:: FeatureStore
   :show-inheritance:
   :members:


TorchWorker
===========

.. currentmodule:: smartsim._core.mli.infrastructure.worker.torch_worker

.. autosummary::

   TorchWorker.load_model
   TorchWorker.transform_input
   TorchWorker.execute
   TorchWorker.transform_output

.. autoclass:: TorchWorker
   :show-inheritance:
   :members:


.. currentmodule:: smartsim._core.mli.infrastructure.worker.worker

InferenceRequest
================

.. autosummary::

   InferenceRequest.has_raw_model
   InferenceRequest.has_model_key
   InferenceRequest.has_raw_inputs
   InferenceRequest.has_input_keys
   InferenceRequest.has_output_keys
   InferenceRequest.has_input_meta

.. autoclass:: InferenceRequest
   :show-inheritance:
   :members:


InferenceReply
===============

.. autosummary::

   InferenceReply.__init__
   InferenceReply.has_outputs
   InferenceReply.has_output_keys

.. autoclass:: InferenceReply
   :show-inheritance:
   :members:


LoadModelResult
===============

.. autosummary::

   LoadModelResult.__init__

.. autoclass:: LoadModelResult
   :show-inheritance:
   :members:


TransformInputResult
====================

.. autosummary::

   TransformInputResult.__init__

.. autoclass:: TransformInputResult
   :show-inheritance:
   :members:


ExecuteResult
=============

.. autosummary::

   ExecuteResult.__init__

.. autoclass:: ExecuteResult
   :show-inheritance:
   :members:


FetchInputResult
================

.. autosummary::

   FetchInputResult.__init__

.. autoclass:: FetchInputResult
   :show-inheritance:
   :members:


TransformOutputResult
======================

.. autosummary::

   TransformOutputResult.__init__

.. autoclass:: TransformOutputResult
   :show-inheritance:
   :members:


CreateInputBatchResult
======================

.. autosummary::

   CreateInputBatchResult.__init__

.. autoclass:: CreateInputBatchResult
   :show-inheritance:
   :members:


FetchModelResult
================

.. autosummary::

   FetchModelResult.__init__

.. autoclass:: FetchModelResult
   :show-inheritance:
   :members:


RequestBatch
============

.. autosummary::

   RequestBatch.__init__
   RequestBatch.has_valid_requests
   RequestBatch.has_raw_model
   RequestBatch.raw_model
   RequestBatch.input_keys
   RequestBatch.output_keys


.. autoclass:: RequestBatch
   :show-inheritance:
   :members:


MachineLearningWorkerCore
=========================

.. autosummary::

   MachineLearningWorkerCore.__init__
   MachineLearningWorkerCore.deserialize_message
   MachineLearningWorkerCore.prepare_outputs
   MachineLearningWorkerCore.fetch_model
   MachineLearningWorkerCore.fetch_inputs
   MachineLearningWorkerCore.place_output

.. autoclass:: MachineLearningWorkerCore
   :show-inheritance:
   :members:


MachineLearningWorkerBase
=========================

.. autosummary::

   MachineLearningWorkerBase.__init__
   MachineLearningWorkerBase.load_model
   MachineLearningWorkerBase.transform_input
   MachineLearningWorkerBase.execute
   MachineLearningWorkerBase.transform_output


.. autoclass:: MachineLearningWorkerBase
   :show-inheritance:
   :members:



.. currentmodule:: smartsim._core.mli.infrastructure.environment_loader

EnvironmentConfigLoader
=======================

.. autosummary::

   EnvironmentConfigLoader.__init__
   EnvironmentConfigLoader.get_backbone
   EnvironmentConfigLoader.get_queue


.. autoclass:: EnvironmentConfigLoader
   :show-inheritance:
   :members:


MessageHandler
==============

.. currentmodule:: smartsim._core.mli.message_handler

.. autosummary::

   MessageHandler.build_tensor_descriptor
   MessageHandler.build_output_tensor_descriptor
   MessageHandler.build_tensor_key
   MessageHandler.build_model
   MessageHandler.build_model_key
   MessageHandler.build_torch_request_attributes
   MessageHandler.build_tf_request_attributes
   MessageHandler.build_torch_response_attributes
   MessageHandler.build_tf_response_attributes
   MessageHandler.build_request
   MessageHandler.serialize_request
   MessageHandler.deserialize_request
   MessageHandler.build_response
   MessageHandler.serialize_response
   MessageHandler.deserialize_response

.. autoclass:: MessageHandler
   :show-inheritance:
   :members:

