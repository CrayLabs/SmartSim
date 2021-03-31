****************
Install SmartSim
****************

Prerequisites
-------------
The prerequisites to begin building SmartSim are:

- Python 3.7.x (or later)
- C compiler
- C++ compiler
- CMake 3.10.x (or later)
- GCC > 5
- GNU Make > 4.0
- autoconf
- automake
- libtool

Optional for building documentation:

- Doxygen
- breathe 4.27.0


Installing From Source
======================

If your system does not include SmartSim
as a module, the instructions in this section
can be used to build and install SmartSim from source.

Note: Windows is not supported and there are currently
no plans to support windows.


Requirements
----------------

Most SmartSim library requirements can be installed via Python's
package manager ``pip`` with the following command executed in the
top level directory of SmartSim.

.. code-block:: bash

  conda activate env # optionally activate virtual env (recommended)
  pip install -r requirements.txt
  pip install -r requirements-dev.txt (optionally download all dev requirements)

If users wish to install these requirements manually, the packages
installed by the previous *pip* command are shown below.  Care
should be taken to match package version numbers to avoid build
and runtime errors.


Local Build (Quick Start)
-------------------------
.. code-block:: bash

  #clone repository
  git clone https://github.com/CrayLabs/SmartSim.git
  #navigate to top-level directory
  cd SmartSim
  #make dependencies for SmartSim
  make deps          # default

  #after dependancies complete successfully, build clients
  make silc

Lastly, after SILC is installed, make sure to reset SmartSim
so that it is aware that the clients have been installed.
Perform this step from the top level of the SmartSim
directory.

.. code-block:: bash

  # in the terminal you are working in
  source setup_env.sh

In addition to installing packages, ``setup_env.sh`` sets
system environment variables that are used by SmartSim
to run experiments and can be used by the user to
locate files needed to  build SmartSim clients into their
applications.


Documentation
-------------

Users can optionally build documentation of SmartSim

.. code-block:: bash

  # in the terminal you are working in
  make docs

Once the documentation has successfully built, users can open the
main documents page from ``doc/_build/html/index.html``





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


