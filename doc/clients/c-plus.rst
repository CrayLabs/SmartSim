***
C++
***

Using the C++ Client
====================

The SmartSim C++ client allows users to send data to and receive
data from other SmartSim entities.  The code snippet below shows
the code required to send and receive data with the C++ client.
To use the C++ client in your simulation code,
simply include ``client.h`` in the source or header files
that leverage the client, and add the ``client.h`` and
``client.cc`` to your simulation code build process.

In the following subsections, general groups of functions
that are provided by the C++ client API will be described.

.. code-block:: c++
  :linenos:

  #include "client.h"

  // Create your simulation data
  double outgoing_data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double incoming_data[5];
  int dims[1] = {5};
  int n_dims = 1;
  std::string key = "data_key";

  // Send your data
  SmartSimClient client;
  client.put_array_double(key.c_str(), outgoing_data, dims, n_dims);

  // Retrieve your data
  client.get_array_double(key.c_str(), incoming_data, dims, n_dims);

Client Initialization
_____________________

The C++ client connection is initialized with the object constructor.
Client connection settings are managed by SmartSim through
environment variables, and as a result, no arguments need to be
passed to the constructor to establish this connection.

Sending and Receiving Data
__________________________

Sending and receiving data is accomplished with a single function call,
respectively. To send data, the C++ client has a set of methods
beginning with ``put_array`` and ``put_scalar`` that are suffixed by
the data type being sent.  Similarly, to retrieve data, the C++
client has a set of methods beginning with ``get_array`` and ``get_scalar``
that are suffixed by the data type being received.

Client Queries
___________________________

The C++ client provides additional methods for checking key existence
and polling functions to block execution until a key exists in the
SmartSim experiment database.  Additionally, functions are provided
to block execution until the value associated with a key is equal
to a user-specified value.

C++ client API
==============

.. doxygenindex::
        :project: cpp_client
