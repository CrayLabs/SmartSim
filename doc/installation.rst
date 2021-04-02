****************
Install SmartSim
****************

There are 3 ways to install SmartSim:
 1. Quick Start - local build for laptops and workstations
 2. Full Install - Full build of SmartSim and SmartRedis
 3. From Source - For SmartSim developers and contributors

For users looking for a full installation of all machine learning
backends and GPU compatibility, see the Full Installation section below.

For users of SmartSim on large cluster or supercomputer systems,
we recommend the full installation. Before install, reference the
launcher documentation (link) to ensure SmartSim has compatibility
with the workload manager of your sites system.

For contributors and developers, we recommend the From Source
installation below. The source build will be able to create docs,
run tests, and includes helpful developer tooling.

=============
Prerequisites
=============

The prerequisites to begin building SmartSim are:

- Python 3.7.x (or later) and pip
- C compiler
- C++ compiler
- CMake 3.10.x (or later)
- GCC > 5
- GNU Make > 4.0
- autoconf
- automake
- libtool

For most developer systems, many of these packages will already
be installed.

.. note::

   For Mac OS users, the version of ``make`` that comes with
   the Mac commandline tools is 3.81 which needs to be updated to install
   SmartSim. Users can ``brew install make`` to get ``make`` > 4.0 but
   brew will install it as ``gmake``. An easy way around this
   is to do ``alias make=gmake``.

.. note::

   Windows is not supported and there are currently no plans
   to support windows.

------------------------------------------------------------

===========
Quick Start
===========

The following is a quick start installation for users looking
to quickly get up and running on their local machine. This
build will install SmartSim for use with Pytorch and TensorFlow
on CPU.


Install SmartSim
================

Follow the steps below for local builds on Mac OS. These instructions
also work for most linux flavors provided the above prerequisites
have been installed.

First, get and unpack the release tarball for SmartSim

.. code-block:: bash

    wget <tarball location>
    tar -xf smartsim-0.3.0.tar.gz

Create a virtual environment and install ``git-lfs``

.. code-block:: bash

    conda create --name=smartsim python=3.7.7
    conda activate smartsim
    conda install git-lfs (if you dont have git-lfs already)
    git-lfs install

Make SmartSim dependencies (see make third-party libraries below).


.. code-block:: bash

    cd smartsim-0.3.0
    make deps # have to have internet access on system


Install SmartSim into your virtual environment through pip, in this
case pip is provided by Conda.

.. code-block:: bash

    pip install -e .

Install the SmartSim user configuration file in your ``$HOME``
directory under ``~/.smartsim/config.toml``. This can also
be installed in other locations. For details, see the full
installation.

.. code-block:: bash

    mkdir ~/.smartsim && cd ~/.smartsim
    touch config.toml

The configuration file specifies user and developer settings
for SmartSim as well as the location of third-party libraries.
If you followed the instructions above, the third-party libraries
will be installed in ``smartsim/third-party/``.

You can copy paste, the below configuration file into the
``config.toml`` file you just created. The only part that
should be changed is the path to where you installed
SmartSim

.. code-block:: toml

    [smartsim]
    # options are "error", "info", and "debug"
    log_level = "info"

    [redis]
    # path to where "redis-server" and "redis-cli" binaries are located
    exe = "/REPLACE/ME/smartsim-0.3.0/third-party/redis/src/redis-server"
    cli = "/REPLACE/ME/smartsim-0.3.0/third-party/redis/src/redis-cli"

    # optional!
    config = "/REPLACE/ME/smartsim-0.3.0/smartsim/database/redis6.conf"

      [redis.ai]
      # path to the redisai "install_cpu" or "install_gpu" dir
      device = "cpu" # cpu or gpu
      install_path = "/REPLACE/ME/smartsim-0.3.0/third-party/RedisAI/install-cpu/"

      [redis.ip]
      # path to build dir for RedisIP
      install_path = "/REPLACE/ME/smartsim-0.3.0/third-party/RedisIP/build/"

    [test]
    # optional!
    launcher = "local"

Done! Now you can follow the tutorial section for working through
Smartsim. If you want to use the SmartRedis Clients in addition to
SmartSim, see below.

Install SmartRedis
==================

The following will show how to install SmartRedis Python client
for use in SmartSim. Note that only the Python client will be built
and the C, C++, and Fortran client will solely be downloaded. For
building a static library of the compiled SmartRedis clients, see
the instructions for the full installation of SmartRedis.

.. include:: ../smartredis/doc/install/python_client.rst

-------------------------------------------------------------------

=================
Full Installation
=================

The full installation of SmartSim includes

  - SmartSim with GPU support and full ML library support
  - SmartRedis Python client
  - SmartRedis static lib for C, C++, and Fortran

Installation Variants
=====================

The following two sections detail how to install variants of SmartSim
for GPU and CPU along with varying levels of support for Machine Learning
libaries. The full install steps (link) will use the GPU build
with all possible backends.


Third Party Libraries
---------------------

`Redis`_,  RedisAI_, and RedisIP_ are required
in order to use all features of SmartSim. Note that if
you are soley using SmartSim for it's launching capabilties
and not utilizing the ``Orchestrator``, these dependencies
do not need to be installed.

There are 4 built-in builds for different types of systems.

	1. default (builds SmartSim backends for Pytorch and TF on CPU)
	2. GPU     (builds SmartSim backends for Pytorch and TF on GPU)
	3. CPU all (builds SmartSim backends for Pytorch, TF, TF-Lite, and Onnx for CPU)
	4. GPU all (builds SmartSim backends for Pytorch, TF, TF-Lite, and Onnx for GPU)

.. _RedisIP: https://github.com/Spartee/RedisIP
.. _Redis: https://github.com/redis/redis
.. _RedisAI: https://github.com/RedisAI/RedisAI

These packages can be downloaded, compiled, and installed
by substituting ONE of the following commands for the ``make deps``
step in the full installation instructions below this section.

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


Install SmartSim for GPU
========================

These instructions will detail the SmartSim for NVIDIA GPU
installation. This can easily be changed for CPU only machine
by changing the ``make-deps`` line below as detailed in the
installation variants section above.

The following steps are to install SmartSim and SmartRedis
with the full feature set and support for TensorFlow, Pytorch,
TensorFlow-Lite, and ONNX runtimes.

.. note::

   This intall requires internet access. Installing on compute
   or MOM nodes of a cluster or supercomputer without public
   internet access will not work.


.. note::

   Currently, SmartSim is solely compatible with NVIDIA GPUs
   and ``CUDA >= 10.2`` is required to build.


Get the SmartSim Release
------------------------

First, get and unpack the release tarball for SmartSim

.. code-block:: bash

    wget <tarball location>
    tar -xf smartsim-0.3.0.tar.gz

Create a virtual environment and install ``git-lfs``

.. code-block:: bash

    conda create --name=smartsim python=3.7.7
    conda activate smartsim
    conda install git-lfs (if you dont have git-lfs already)
    git-lfs install

Setup CUDA for Install
----------------------

Next, install (or module load) and set the paths to the NVIDIA CUDA
libraries. For large systems, this is often already installed.

If you plan to run SmartSim on GPU you will need
to follow the steps below. Otherwise if you plan to run solely
on CPU, skip this subsection.

For users of systems with modules that include CUDA, like most
Cray systems with GPUs. Do the following

.. code-block:: bash

  # Skip in favor of module load if cudnn is a module
  conda install cudnn=7.6.5

  module load cudatoolkit

  # and make sure the following variables are set in your env
  # after module load. check with ``env | grep CUD``
  export CUDA_HOME=/path/to/cuda
  export CUDNN_LIBRARY=/path/to/cudnn-7.6.5-cuda10.2_0/lib
  export CUDNN_INCLUDE_DIR=/path/to/cudnn-7.6.5-cuda10.2_0/include

For systems without CUDA installed, install CUDA > 10.2 prior
to this setup.

If you did the above step, you don't have to do this step.
If not, make sure to change the paths to the CUDNN and CUDA libraries.

.. code-block:: bash

  # Install CUDA requirements
  conda install cudatoolkit=10.2 cudnn=7.6.5
  export CUDA_HOME=/path/to/cuda
  export CUDNN_LIBRARY=/path/to/miniconda/pkgs/cudnn-7.6.5-cuda10.2_0/lib
  export CUDNN_INCLUDE_DIR=/path/to/miniconda/pkgs/cudnn-7.6.5-cuda10.2_0/include

Install all SmartSim Dependencies
----------------------------------

Make SmartSim dependencies (see make third-party libraries below).
If on a Cray system, be sure to set the correct toolchain. SmartSim
is tested on ``PrgEnv-GNU`` and ``PrgEnv-Cray`` modules.

.. note::

    If on a Cray, please note that the intel and PGI compiler
    toolchains are currently not supported by SmartSim.

.. code-block:: bash

    cd smartsim
    export CRAYPE_LINK_TYPE=dynamic # (optional only for cray systems)
    module load PrgEnv-Gnu          # or PrgEnv-Cray (optional only for cray systems)

    make deps-gpu-all

    # or for CPU-only (see installation variants section above)
    make deps-cpu-all

Keep in mind, the libraries installed above need to be accessable
by SmartSim at runtime. If using a networked file system (NFS),
make sure to install these somewhere reachable from head, MOM, and
compute nodes.

Install SmartSim into Python Environment
----------------------------------------

Install SmartSim into your virtual environment through pip, in this
case pip is provided by Conda.

.. code-block:: bash

    pip install -e .
    # or if you want all the dev dependencies which includes Pytorch 1.7
    pip install -e .[dev] #

Setup SmartSim Configuration File
----------------------------------

Install the SmartSim user configuration file.
Usually this is in your ``$HOME`` directory under
``~/.smartsim/config.toml``, but for users of networked filesystems
where the ``$HOME`` directory is not mounted, do the following

.. code-block:: bash

    # set the configuration directory to somewhere accessable
    # (e.g. /lustre) for HPC systems

    echo "export SMARTSIM_HOME=/REPLACE/ME/.smartsim" >> ~/.bashrc
    cd && source .bashrc
    mkdir $SMARTSIM_HOME && cd $SMARTSIM_HOME
    touch config.toml

The configuration file specifies user and developer settings
for SmartSim as well as the location of third-party libraries.
If you followed the instructions above, the third-party libraries
will be installed in ``smartsim/third-party/``.

You can copy paste, the below configuration file into the
``config.toml`` file you just created. The only part that
should be changed is the path to where you installed
SmartSim

.. code-block:: toml

    [smartsim]
    # options are "error", "info", and "debug"
    log_level = "info"

    [redis]
    # path to where "redis-server" and "redis-cli" binaries are located
    exe = "/REPLACE/ME/smartsim-0.3.0/third-party/redis/src/redis-server"
    cli = "/REPLACE/ME/smartsim-0.3.0/third-party/redis/src/redis-cli"

    # optional!
    config = "/REPLACE/ME/smartsim-0.3.0/smartsim/database/redis6.conf"

      [redis.ai]
      # path to the redisai "install_cpu" or "install_gpu" dir
      device = "cpu" # cpu or gpu
      install_path = "/REPLACE/ME/smartsim-0.3.0/third-party/RedisAI/install-cpu/"

      [redis.ip]
      # path to build dir for RedisIP
      install_path = "/REPLACE/ME/smartsim-0.3.0/third-party/RedisIP/build/"

    [test]
    # optional!
    launcher = "local"

Install SmartRedis
==================

Now that the infrastructure library is installed, we complete the full
SmartSim installation by installing the SmartRedis clients.

We will build two components:

  1. The SmartRedis Python Client
  2. The SmartRedis C, C++, and Fortran Clients as a static library


.. include:: ../smartredis/doc/install/python_client.rst


Build SmartRedis as a Static Library
-------------------------------------


.. include:: ../smartredis/doc/install/lib.rst


--------------------------------------------------------------------


===========
From Source
===========

Install SmartSim from Source
============================

To install SmartSim from source, simply clone the github repo
at ``https://github.com/CrayLabs/SmartSim`` and follow the
instructions for the full installation.

From source, users have the option of downloading SmartRedis
using the Makefile command ``make smartredis``.


Building the Documentation
==========================

.. note::
    To build the full documentation, user need to install
    ``doxygen 1.9.1``. For Mac OS users, doxygen can be
    installed through ``brew install doxygen``

Users can optionally build documentation of SmartSim

.. code-block:: bash

  make smartredis   # (need the docs from SmartRedis as well)
  cd /smartsim      # top level smartsim dir
  make docs

Once the documentation has successfully built, users can open the
main documents page from ``doc/_build/html/index.html``





Install SmartRedis from Source
==============================

.. include:: ../smartredis/doc/install/from_source.rst

