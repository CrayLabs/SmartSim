************
Orchestrator
************


The ``Orchestrator`` is an in-memory database that is launched prior to all other
entities within an ``Experiment``. The ``Orchestrator`` can be used to store and retrieve
data during the course of an experiment. In order to stream data into
or receive data from the ``Orchestrator``, one of the SmartSim clients (SmartRedis) has to be
used within a Model.

.. |orchestrator| image:: images/Orchestrator.png
  :width: 700
  :alt: Alternative text

|orchestrator|

Combined with the SmartRedis clients, the ``Orchestrator`` is capable of hosting and executing
AI models written in Python on CPU or GPU. The ``Orchestrator`` supports models written with
TensorFlow, Pytorch, TensorFlow-Lite, or models saved in an ONNX format (e.g. sci-kit learn).


The :ref:`Orchestrator API <local_orc_api>` is implemented for each launcher that
SmartSim supports.

 - :ref:`SlurmOrchestrator <slurm_orc_api>` for Slurm managed systems
 - :ref:`CobaltOrchestrator <cobalt_orc_api>` for Cobalt managed systems
 - :ref:`PBSOrchestrator <pbs_orc_api>` for PBSPro managed systems.
 - :ref:`LSFOrchestrator <lsf_orc_api>` for LSF managed systems.

The base ``Orchestrator`` class can be used for launching Redis
locally on single node workstations or laptops.


Redis
=====

.. _Redis: https://github.com/redis/redis
.. _RedisAI: https://github.com/RedisAI/RedisAI

The ``Orchestrator`` is built on `Redis`_. Largely, the job of the ``Orchestartor`` is to
create a Python reference to a Redis deployment so that users can launch, monitor
and stop a Redis deployment on workstations and HPC systems.

Redis was chosen for the Orchestrator because it resides in-memory, can be distributed on-node
as well as across nodes, and provides low-latency data access to many clients in parallel. The
Redis ecosystem was a primary driver as the Redis module system provides APIs for languages,
libraries, and techniques used in Data Science. In particular, the ``Orchestrator``
relies on `RedisAI`_ to provide access to Machine Learning runtimes.

At its core, Redis is a key-value store. This means that put/get semantics are used to send
messages to and from the database. SmartRedis clients use a specific hashing algorithm, CRC16, to ensure
that data is evenly distributed amongst all database nodes. Notably, a user is not required to
know where (which database node) data or Datasets (see Dataset API) are stored as the
SmartRedis clients will infer their location for the user.

