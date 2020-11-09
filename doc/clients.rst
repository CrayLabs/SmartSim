
****
SILC
****

The SmartSim Infrastructure Library Clients are essentially
Redis clients with additional functionality. In particular,
the SILC clients allow a user to send and receive n-dimensional
``Tensors`` with metadata, a crucial data format for data
analysis and machine learning.

Each client in SILC has distributed support for Redis clusters
and can work with both Redis and KeyDB.

Furthermore, the client implementations in SILC are all
RedisAI compatable meaning that they can directly set
and run Machine Learning and Deep Learning models stored
within a Redis database.


.. list-table:: Supported Languages
   :widths: 25 25 25
   :header-rows: 1
   :align: center

   * - Language
     - Version/Standard
     - Status
   * - Python
     - 3.7+
     - In Development
   * - C++
     - C++11
     - Stable
   * - C
     - C99
     - In Development
   * - Fortran
     - Fortran 2003 +
     - Awaiting Development


Simulation and data analytics codes communicate with the database using
SmartSim clients written in the native language of the codebase. These
clients perform two essential tasks (both of which are opaque to the application):

 1. Serialization/deserialization of data
 2. Communication with the database

The API for these clients are designed so that implementation within
simulation and analysis codes requires minimal modification to the underlying
codebase.


.. |SmartSim Clients| image:: images/Smartsim_Client_Communication.png
  :width: 500
  :alt: Alternative text

|SmartSim Clients|


