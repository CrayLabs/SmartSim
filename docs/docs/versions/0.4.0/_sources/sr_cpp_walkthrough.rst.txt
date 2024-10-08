***
C++
***


In this section, examples are presented using the SmartRedis C++
API to interact with the RedisAI tensor, model, and script
data types.  Additionally, an example of utilizing the
SmartRedis ``DataSet`` API is also provided.



.. note::
      The C++ API examples rely on the ``SSDB`` environment
      variable being set to the address and port of the Redis database.


.. note::
    The C++ API examples are written
    to connect to a clustered database or clustered SmartSim Orchestrator.
    Update the ``Client`` constructor ``cluster`` flag to `false`
    to connect to a single shard (single compute host) database.




Tensors
=======

The following example shows how to send a receive a tensor using the
SmartRedis C++ client API.

.. literalinclude:: ../smartredis/examples/serial/cpp/smartredis_put_get_3D.cpp
  :linenos:
  :language: C++

DataSets
========

The C++ client can store and retrieve tensors and metadata in datasets.
For further information about datasets, please refer to the :ref:`Dataset
section of the Data Structures documentation page <data_structures_dataset>`.

The code below shows how to store and retrieve tensors and metadata
which belong to a ``DataSet``.

.. literalinclude:: ../smartredis/examples/serial/cpp/smartredis_dataset.cpp
  :linenos:
  :language: C++

.. _SR CPP Models:


Models
======

The following example shows how to store, and use a DL model
in the database with the C++ Client.  The model is stored as a file
in the ``../../../common/mnist_data/`` path relative to the
compiled executable.  Note that this example also sets and
executes a preprocessing script.

.. literalinclude:: ../smartredis/examples/serial/cpp/smartredis_model.cpp
  :linenos:
  :language: C++

.. _SR CPP Scripts:

Scripts
=======

The example in :ref:`SR CPP Models` shows how to store, and use a PyTorch script
in the database with the C++ Client.  The script is stored a file
in the ``../../../common/mnist_data/`` path relative to the
compiled executable.  Note that this example also sets and
executes a PyTorch model.

.. _SR CPP Parallel MPI:

Parallel (MPI) execution
========================

In this example, the example shown in :ref:`SR CPP Models` and
:ref:`SR CPP Scripts` is adapted to run in parallel using MPI.
This example has the same functionality, however,
it shows how keys can be prefixed to prevent key
collisions across MPI ranks.  Note that only one
model and script are set, which is shared across
all ranks.

For completeness, the pre-processing script
source code is also shown.

**C++ program**

.. literalinclude:: ../smartredis/examples/parallel/cpp/smartredis_mnist.cpp
  :linenos:
  :language: C++

**Python Pre-Processing**

.. literalinclude:: ../smartredis/examples/common/mnist_data/data_processing_script.txt
  :linenos:
  :language: Python
  :lines: 15-20

