
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
     - In Development


For more information on the SILC clients, please refer to the SILC documentation
