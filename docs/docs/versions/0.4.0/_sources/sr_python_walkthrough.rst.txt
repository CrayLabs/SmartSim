
******
Python
******

This section will detail the SmartRedis Python client and how to
use it within SmartSim applications and RedisAI in general.


.. note::
      The Python API examples are written to connect to a
      database at ``127.0.0.1:6379``.  When running this example,
      ensure that the address and port of your Redis instance are used.



Tensors
=======

The Python client has the ability to send and receive tensors from
the Redis database.  The tensors are stored in the Redis database
as RedisAI data structures.  Additionally, Python client API
functions involving tensor data are compatible with Numpy arrays
and do not require any other data types.

.. literalinclude:: ../smartredis/examples/serial/python/example_put_get_tensor.py
  :language: python
  :linenos:
  :lines: 26-40


Datasets
========

The Python client can store and retrieve tensors and metadata in datasets.
For further information about datasets, please refer to the :ref:`Dataset
section of the Data Structures documentation page <data_structures_dataset>`.

The code below shows how to store and retrieve tensors which belong to a ``DataSet``.

.. literalinclude:: ../smartredis/examples/serial/python/example_put_get_dataset.py
  :language: python
  :linenos:
  :lines: 26-52

Models
======

The SmartRedis clients allow users to set and use a PyTorch, ONNX, TensorFlow,
or TensorFlow Lite model in the database. Models can be sent to the database directly
from memory or from a file. The code below illustrates how a
jit-traced PyTorch model can be used with the Python client library.

.. literalinclude:: ../smartredis/examples/serial/python/example_model_torch.py
  :language: python
  :linenos:
  :lines: 26-71

Models can also be set from a file, as in the code below.

.. literalinclude:: ../smartredis/examples/serial/python/example_model_file_torch.py
  :language: python
  :linenos:
  :lines: 26-69

Scripts
=======

Scripts are a way to store python-executable code in the database. The Python
client can send scripts to the dataset from a file, or directly from memory.

As an example, the code below illustrates how a function can be defined and sent
to the database on the fly, without storing it in an intermediate file.

.. literalinclude:: ../smartredis/examples/serial/python/example_script.py
  :language: python
  :linenos:
  :lines: 26-66

The code below shows how to set a script from a file.  Running the
script set from file uses the same API calls as the example shown
above.

.. literalinclude:: ../smartredis/examples/serial/python/example_script_file.py
  :language: python
  :linenos:
  :lines: 26-41

The content of the script file has to be written
in Python. For the example above, the file ``data_processing_script.txt``
looks like this:

.. literalinclude:: ../smartredis/examples/serial/python/data_processing_script.txt
  :language: python
  :linenos:

