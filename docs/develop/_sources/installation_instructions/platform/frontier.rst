OLCF Frontier
=============

Known limitations
-----------------

We are continually working on getting all the features of SmartSim working on
Frontier, however we do have some known limitations:

* For now, only Torch models are supported. If you need Tensorflow or ONNX
  support please contact us
* All SmartSim experiments must be run from Lustre, _not_ your home directory
* The colocated database will fail without specifying ``custom_pinning``. This
  is because the default pinning assumes that processor 0 is available, but the
  'low-noise' default on Frontier reserves the processor on each NUMA node.
  Users should pass a list of processor ids to the ``custom_pinning`` argument that
  avoids the reserved processors
* The Singularity-based tests are currently failing. We are investigating how to
  interact with Frontier's configuration. Please contact us if this is interfering
  with your application

Please raise an issue in the SmartSim Github or contact the developers if the above
issues are affecting your workflow or if you find any other problems.

One-time Setup
--------------

To install the SmartRedis and SmartSim python packages on Frontier, please follow
these instructions, being sure to set the following variables

.. code:: bash

   export PROJECT_NAME=CHANGE_ME

**Step 1:** Create and activate a virtual environment for SmartSim:

.. code:: bash

   module load PrgEnv-gnu miniforge3 rocm/6.1.3

   export SCRATCH=/lustre/orion/$PROJECT_NAME/scratch/$USER/
   conda create -n smartsim python=3.11
   source activate smartsim

**Step 2:** Build the SmartRedis C++ and Fortran libraries:

.. code:: bash

   cd $SCRATCH
   git clone https://github.com/CrayLabs/SmartRedis.git
   cd SmartRedis
   make lib-with-fortran
   pip install .

**Step 3:** Install SmartSim in the conda environment:

.. code:: bash

   cd $SCRATCH
   pip install git+https://github.com/CrayLabs/SmartSim.git

**Step 4:** Build Redis, RedisAI, the backends, and all the Python packages:

.. code:: bash

   smart build --device=rocm-6

**Step 5:** Check that SmartSim has been installed and built correctly:

.. code:: bash

   # Optimizations for inference
   export MIOPEN_USER_DB_PATH="/tmp/${USER}/my-miopen-cache"
   export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
   rm -rf $MIOPEN_USER_DB_PATH
   mkdir -p $MIOPEN_USER_DB_PATH

   # Run the install validation utility
   smart validate --device gpu

The following output indicates a successful install:

.. code:: bash

   [SmartSim] INFO Verifying Tensor Transfer
   [SmartSim] INFO Verifying Torch Backend
   16:26:35 login SmartSim[557020:MainThread] INFO Success!

Post-installation
-----------------

Before running SmartSim, the environment should match the one used to
build, and some variables should be set to optimize performance:

.. code:: bash

   # Set these to the same values that were used for install
   export PROJECT_NAME=CHANGE_ME

.. code:: bash

   module load PrgEnv-gnu miniforge3 rocm/6.1.3
   source activate smartsim

   # Optimizations for inference
   export MIOPEN_USER_DB_PATH="/tmp/${USER}/my-miopen-cache"
   export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
   rm -rf ${MIOPEN_USER_DB_PATH}
   mkdir -p ${MIOPEN_USER_DB_PATH}

Binding DBs to Slingshot
------------------------

Each Frontier node has *four* NICs, which also means users need to bind
DBs to *four* network interfaces, ``hsn0``, ``hsn1``, ``hsn2``,
``hsn3``. Typically, orchestrators will need to be created in the
following way:

.. code:: python

   exp = Experiment("my_exp", launcher="slurm")
   orc = exp.create_database(db_nodes=3, interface=["hsn0","hsn1","hsn2","hsn3"], single_cmd=True)
