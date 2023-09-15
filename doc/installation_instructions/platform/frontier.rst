OLCF Frontier
=============

Summary
-------

Frontier is an AMD CPU/AMD GPU system.

As of 2023-07-06, users can use the following instructions, however we
anticipate that all the SmartSim dependencies will be available system-wide via
the modules system.

Known limitations
-----------------

We are continually working on getting all the features of SmartSim working on
Frontier, however we do have some known limitations:

* For now, only Torch models are supported. We are working to find a recipe to
  install Tensorflow with ROCm support from scratch
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

Build process
-------------

To install the SmartRedis and SmartSim python packages on Frontier, please follow
these instructions, being sure to set the following variables

.. code:: bash

   export PROJECT_NAME=CHANGE_ME
   export VENV_NAME=CHANGE_ME

Then continue with the install:

.. code:: bash

   module load PrgEnv-gnu-amd git-lfs cmake cray-python
   module unload xalt amd-mixed
   module load rocm/4.5.2
   export CC=gcc
   export CXX=g++

   export SCRATCH=/lustre/orion/$PROJECT_NAME/scratch/$USER/
   export VENV_HOME=$SCRATCH/$VENV_NAME/

   python3 -m venv $VENV_HOME
   source $VENV_HOME/bin/activate
   pip install torch==1.11.0+rocm4.5.2 torchvision==0.12.0+rocm4.5.2 torchaudio==0.11.0  --extra-index-url  https://download.pytorch.org/whl/rocm4.5.2


   cd $SCRATCH
   git clone https://github.com/CrayLabs/SmartRedis.git
   cd SmartRedis
   make lib-with-fortran
   pip install .

   # Download SmartSim and site-specific files
   cd $SCRATCH
   git clone https://github.com/CrayLabs/site-deployments.git
   git clone https://github.com/CrayLabs/SmartSim.git
   cd SmartSim
   pip install -e .[dev]

Next to finish the compilation, we need to manually modify one of the auxiliary
cmake files that comes packaged with Torch

.. code:: bash

   export TORCH_CMAKE_DIR=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
   # Manual step: modify all references to the 'rocm' directory to rocm-4.5.2
   vim $TORCH_CMAKE_DIR/Caffe2/Caffe2Targets.cmake

Finally, build Redis (or keydb for a more performant solution), RedisAI, and the
machine-learning backends using:

.. code:: bash

   KEYDB_FLAG="" # set this to --keydb if desired
   smart build --device gpu --torch_dir $TORCH_CMAKE_DIR --no_tf -v $(KEYDB_FLAG)

Set up environment
------------------

Before running SmartSim, the environment should match the one used to
build, and some variables should be set to work around some ROCm PyTorch
issues:

.. code:: bash

   # Set these to the same values that were used for install
   export PROJECT_NAME=CHANGE_ME
   export VENV_NAME=CHANGE_ME

.. code:: bash

   module load PrgEnv-gnu-amd git-lfs cmake cray-python
   module unload xalt amd-mixed
   module load rocm/4.5.2

   export SCRATCH=/lustre/orion/$PROJECT_NAME/scratch/$USER/
   export MIOPEN_USER_DB_PATH=/tmp/miopendb/
   export MIOPEN_SYSTEM_DB_PATH=$MIOPEN_USER_DB_PATH
   mkdir -p $MIOPEN_USER_DB_PATH
   export MIOPEN_DISABLE_CACHE=1
   export VENV_HOME=$SCRATCH/$VENV_NAME/
   source $VENV_HOME/bin/activate
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VENV_HOME/lib/python3.9/site-packages/torch/lib

Binding DBs to Slingshot
------------------------

Each Frontier node has *four* NICs, which also means users need to bind
DBs to *four* network interfaces, ``hsn0``, ``hsn1``, ``hsn2``,
``hsn3``. Typically, orchestrators will need to be created in the
following way:

.. code:: python

   exp = Experiment("my_exp", launcher="slurm")
   orc = exp.create_database(db_nodes=3, interface=["hsn0","hsn1","hsn2","hsn3"], single_cmd=True)

Running tests
-------------

The same environment set to run SmartSim must be set to run tests. The
environment variables needed to run the test suite are the following:

.. code:: bash

   export SMARTSIM_TEST_ACCOUNT=PROJECT_NAME # Change this to above
   export SMARTSIM_TEST_LAUNCHER=slurm
   export SMARTSIM_TEST_DEVICE=gpu
   export SMARTSIM_TEST_PORT=6789
   export SMARTSIM_TEST_INTERFACE="hsn0,hsn1,hsn2,hsn3"
