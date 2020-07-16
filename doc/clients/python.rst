******
Python
******

Using the Python Client
=======================

The SmartSim Python client allows users to send data to and receive data from
other SmartSim entities stored in the database. The code snippet below shows
the code required to send and receive data with the Python client. In the
following subsections, general groups of functions that are provided by the
Python client API will be described.

.. code-block:: python
  :linenos:

  from smartsim import Client
  import numpy as np

  outgoing_data = np.asarray([[2.0,2.0,1.0],[2.0,3.0,4.0]])
  client = Client(cluster=True)
  client.put_array_nd_float64("synthetic_data", outgoing_data)
  incoming_data = client.get_array_nd_float64("synthetic_data")

Client Initialization
---------------------

The Python client connection is initialized with the object constructor.
The optional boolean argument ``cluster`` indicates whether the client
will be connecting to a single database node or multiple distributed
nodes which is referred to as a cluster.

Sending and Receiving Data
--------------------------

Sending and receiving data is accomplished with a single function call,
respectively. To send data, the Python client has a set of methods
beginning with ``put_array`` and ``put_scalar`` that are suffixed by
the data type being sent.  Similarly, to retrieve data, the Python
client has a set of methods beginning with ``get_array`` and ``get_scalar``
that are suffixed by the data type being received.

Client Queries
--------------

The Python client provides additional methods for checking key existence
and polling functions to block execution until a key exists in the
SmartSim experiment database.  Additionally, functions are provided
to block execution until the value associated with a key is equal
to a user-specified value.

Python Client API
=================

.. automodule:: smartsim.clients.client
   :members:
