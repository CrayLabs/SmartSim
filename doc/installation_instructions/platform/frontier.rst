OLCF Frontier
=============

Known limitations
-----------------

We are continually working on getting all the features of SmartSim working on
Frontier, however we do have some known limitations:

* For now, only Torch and ONNX runtime models are supported. If you need
  Tensorflow support please contact us
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
   export VENV_NAME=CHANGE_ME

**Step 1:** Create and activate a virtual environment for SmartSim:

.. code:: bash

   module load PrgEnv-gnu cray-python
   module load rocm/6.1.3

   export SCRATCH=/lustre/orion/$PROJECT_NAME/scratch/$USER/
   export VENV_HOME=$SCRATCH/$VENV_NAME/

   python3 -m venv $VENV_HOME
   source $VENV_HOME/bin/activate

**Step 2:** Install SmartSim in the conda environment:

.. code:: bash

   cd $SCRATCH
   git clone https://github.com/CrayLabs/SmartRedis.git
   cd SmartRedis
   make lib-with-fortran
   pip install .

   # Download SmartSim and site-specific files
   cd $SCRATCH
   pip install git+https://github.com/CrayLabs/SmartSim.git

**Step 3:** Build Redis, RedisAI, the backends, and all the Python packages:

.. code:: bash

   smart build --device=rocm-6

**Step 4:** Check that SmartSim has been installed and built correctly:

.. code:: bash

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
   export VENV_NAME=CHANGE_ME

.. code:: bash

   module load PrgEnv-gnu
   module load rocm/6.1.3

   # Optimizations for inference
   export SCRATCH=/lustre/orion/$PROJECT_NAME/scratch/$USER/
   export MIOPEN_USER_DB_PATH=/tmp/miopendb/
   export MIOPEN_SYSTEM_DB_PATH=$MIOPEN_USER_DB_PATH
   mkdir -p $MIOPEN_USER_DB_PATH
   export MIOPEN_DISABLE_CACHE=1
   export VENV_HOME=$SCRATCH/$VENV_NAME/
   source $VENV_HOME/bin/activate

Binding DBs to Slingshot
------------------------

Each Frontier node has *four* NICs, which also means users need to bind
DBs to *four* network interfaces, ``hsn0``, ``hsn1``, ``hsn2``,
``hsn3``. Typically, orchestrators will need to be created in the
following way:

.. code:: python

   exp = Experiment("my_exp", launcher="slurm")
   orc = exp.create_database(db_nodes=3, interface=["hsn0","hsn1","hsn2","hsn3"], single_cmd=True)
