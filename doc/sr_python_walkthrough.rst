
******
Python
******

This section details the SmartRedis Python client to demonstrate its general use within SmartSim applications and RedisAI.


.. note::
      The following Python API examples connect to a
      database at the address:port ``127.0.0.1:6379``.  When replicating the following example,
      ensure that you use the address:port of your local Redis instance.



Tensors
=======

The Python client can send and receive tensors from the Redis database,
where they are stored as RedisAI data structures. Additionally, Python client API
functions involving tensor data are compatible with Numpy arrays
and do not require other data types.

.. literalinclude:: ../smartredis/examples/serial/python/example_put_get_tensor.py
  :language: python
  :linenos:
  :lines: 26-40


Datasets
========

The Python client can store and retrieve tensors and metadata in datasets.
For further information about datasets, please refer to the :ref:`Dataset
section of the Data Structures documentation page <data-structures-dataset>`.

The code below shows how to store and retrieve tensors that belong to a ``DataSet``.

.. literalinclude:: ../smartredis/examples/serial/python/example_put_get_dataset.py
  :language: python
  :linenos:
  :lines: 27-51

Models
======

The SmartRedis clients allow users to set and use a PyTorch, ONNX, TensorFlow,
or TensorFlow Lite model in the database. Models can be sent to the database directly
from memory or a file. The code below illustrates how a
jit-traced PyTorch model can be used with the Python client library.

.. literalinclude:: ../smartredis/examples/serial/python/example_model_torch.py
  :language: python
  :linenos:
  :lines: 27-70

Users can set models from a file, as shown in the code below.

.. literalinclude:: ../smartredis/examples/serial/python/example_model_file_torch.py
  :language: python
  :linenos:
  :lines: 27-68

Scripts
=======

Scripts are a way to store python-executable code in the database. The Python
client can send scripts to the dataset from a file or directly from memory.

The code below illustrates how to avoid storing a function in an intermediate file.
With this technique, we can define and send a function to the database on the fly.

.. literalinclude:: ../smartredis/examples/serial/python/example_script.py
  :language: python
  :linenos:
  :lines: 26-66

The code below shows how to set a script from a file. Running the script set from
the file uses the same API calls as in the example shown above.

.. literalinclude:: ../smartredis/examples/serial/python/example_script_file.py
  :language: python
  :linenos:
  :lines: 26-41

This file must be a valid Python script. For the example above, the file ``data_processing_script.txt``
looks like this:

.. literalinclude:: ../smartredis/examples/serial/python/data_processing_script.txt
  :language: python
  :linenos:
