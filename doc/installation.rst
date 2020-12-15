****************
Install SmartSim
****************

From Source
===========

If your system does not include SmartSim
as a module, the instructions in this section
can be used to build and install SmartSim from source.

Prerequisites
-------------
The prerequisites to begin building SmartSim are:

- Python 3.7.x (or later)
- C compiler
- C++ compiler
- CMake 3.10.x (or later)
- GCC > 5

Windows is not supported and there are currently no plans to support windows.

Python libraries
----------------

Most SmartSim library requirements can be installed via Python's
package manager ``pip`` with the following command executed in the
top level directory of SmartSim.

.. code-block:: bash

  conda activate env # optionally activate virtual env (recommended)
  pip install -r requirements.txt
  pip install -r requirements-dex.txt (optionally download all dev requirements)

If users wish to install these requirements manually, the packages
installed by the previous *pip* command are shown below.  Care
should be taken to match package version numbers to avoid build
and runtime errors.


GPU Dependencies
----------------

If you plan to run SmartSim's architecture on GPU you will need
to follow the steps below. Otherwise if you plan to run solely
on CPU, feel free to skip these steps

For this portion you will want CUDA to be installed. Most likely,
CUDA will be installed somewhere like `/usr/local/cuda`

.. code-block:: bash

  # Install CUDA requirements
  conda install cudatoolkit=10.2 cudnn=7.6.5

  # (optional) Load CUDA module instead of using conda
  module load cudatoolkit

  # set cuda and cudnn environment variables
  export CUDNN_LIBRARY=/path/to/miniconda3/pkgs/cudnn-7.6.5-cuda10.2_0/lib
  export CUDNN_INCLUDE_DIR=/path/to/miniconda3/pkgs/cudnn-7.6.5-cuda10.2_0/include
  export CUDATOOLKIT_HOME=/path/to/miniconda3/pkgs/cudatoolkit-10.2.89-hfd86e86_1/


Third Party Libraries
---------------------

KeyDB_, `Redis`_, and RedisAI_ are also required
in order to use all features of SmartSim. These packages
do not require many dependencies, but it is worth checking that
your system meets the prerequisites listed on the project
github page.

There are 4 built-in builds for different types of systems.

	1. default (builds SmartSim backends for Pytorch and TF on CPU)
	2. GPU     (builds SmartSim backends for Pytorch and TF on GPU)
	3. CPU all (builds SmartSim backends for Pytorch, TF, TF-Lite, and Onnx for CPU)
	4. GPU all (builds SmartSim backends for Pytorch, TF, TF-Lite, and Onnx for GPU)

.. _KeyDB: https://github.com/JohnSully/KeyDB
.. _Redis: https://github.com/redis/redis
.. _RedisAI: https://github.com/RedisAI/RedisAI

These packages can be downloaded, compiled, and installed
by executing the following command in the top level of the SmartSim project:


.. code-block:: bash

  # in the top level of the SmartSim directory
  # perform only one of the following
  make deps          # default
  make deps-gpu      # gpu default
  make deps-cpu-all  # all cpu backends
  make deps-gpu-all  # all gpu backends


The ``make deps`` command will install the three packages into
the ``third-party`` directory in the top level directory of
SmartSim.

In addition to installing packages, ``setup_env.sh`` sets
system environment variables that are used by SmartSim
to run experiments and can be used by the user to
locate files needed to  build SmartSim clients into their
applications.  The use of environment variables for compiling
SmartSim clients into applications is discussed in the client
documentation. The user should source ``setup_env.sh`` whenever
beginning a new session to ensure that environment
variables are properly set.

.. code-block:: bash

  source setup_env.sh


Building SILC from Source
=========================

Building the client libraries (SILC) is straightforward from
an existing SmartSim installation. Follow these steps
from the top level of the SmartSim directory

.. code-block:: bash

  # get the source code
  git clone #link to SILC repo#	silc
  git checkout develop # or a release branch

  # build the dependencies
  cd silc
  make deps

  # build the python client (still in top level of silc directory)
  make pyclient

Lastly, after SILC is installed, make sure to reset SmartSim
so that it is aware that the clients have been installed.
Perform this step from the top level of the SmartSim
directory.

.. code-block:: bash

  # in the terminal you are working in
  source setup_env.sh


Suggested Environments for Cray Systems
=======================================

SmartSim has been built and tested on a number of
Cray XC and CS systems.  To help users build from source
on the systems that have been tested, in this section
the terminal commands to load a working environment
will be given.

Osprey (Cray CS)
----------------

.. code-block:: console

		module load PrgEnv-cray/1.0.6
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load cmake/3.10.2
		module load gcc/8.1.0
		module load cudatoolkit/10.1

Loon (Cray CS)
--------------

.. code-block:: console

		module load PrgEnv-cray/1.0.6
		module unload cray-libsci/17.12.1
		module load cmake/3.10.2
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Raptor (Cray CS)
----------------

.. code-block:: console

		module load PrgEnv-cray/1.0.6
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load cmake/3.10.3
		module load gcc/8.1.0

Tiger (Cray XC)
---------------

.. code-block:: console

		module load PrgEnv-cray/6.0.7
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Jupiter (Cray XC)
-----------------

.. code-block:: console

		module load PrgEnv-cray/6.0.7
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Heron
-----

.. code-block:: console

		module load PrgEnv-cray/6.0.7
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Cicero (Cray XC)
----------------

*Default system configurations and modules are sufficient on Cicero.*
