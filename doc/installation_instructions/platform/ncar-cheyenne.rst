
Cheyenne at NCAR
================

Since SmartSim does not currently support the Message Passing Toolkit (MPT),
Cheyenne users of SmartSim will need to utilize OpenMPI.

The following module commands were utilized to run the examples:

.. code-block:: bash

  $ module purge
  $ module load ncarenv/1.3 gnu/8.3.0 ncarcompilers/0.5.0 netcdf/4.7.4 openmpi/4.0.5

With this environment loaded, users will need to build and install both SmartSim
and SmartRedis through pip. Usually we recommend users installing or loading
miniconda and using the pip that comes with that installation.

.. code-block:: bash

  $ pip install smartsim
  $ smart build --device cpu  #(Since Cheyenne does not have GPUs)

To make the SmartRedis library (C, C++, Fortran clients), follow these steps
with the same environment loaded.

.. code-block:: bash

  # clone SmartRedis and build
  $ git clone https://github.com/SmartRedis.git smartredis
  $ cd smartredis
  $ make lib

