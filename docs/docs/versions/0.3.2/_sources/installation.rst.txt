************
Installation
************

The following will show how to install both SmartSim and SmartRedis


===========
Quick Start
===========

If you want to get started with SmartSim quickly, you can install it through pip.
We highly recommend starting a fresh Python environment.

.. code-block:: bash

    $ pip install smartsim
    $ smart --device cpu

This will install SmartSim with PyTorch and TensorFlow backends for the Orchestrator
database as well as the SmartRedis Python client.


==================
Supported Versions
==================


.. list-table:: Supported System for Pre-built Wheels
   :widths: 50 50 50 50
   :header-rows: 1
   :align: center

   * - Platform
     - CPU
     - GPU
     - Python Versions
   * - MacOS
     - x86_64
     - not supported
     - 3.7 - 3.9
   * - Linux
     - x64_64
     - Nvidia
     - 3.7 - 3.9

.. note::

  Windows is not supported and there are currently no plans
  to support Windows.

SmartSim supports multiple machine learning libraries through
the use of RedisAI_. The following libraries are supported.

.. list-table:: Supported ML Libraries
   :widths: 50 50 50 50
   :header-rows: 1
   :align: center

   * - Library
     - Versions
     - Python Versions
     - Built By Default
   * - PyTorch_
     - 1.7
     - 3.7 - 3.9
     - Yes
   * - Tensorflow_ / Keras_
     - 2.4.2
     - 3.7 - 3.8
     - Yes
   * - ONNX_
     - 1.6 - 1.7
     - 3.7 - 3.9
     - No

TensorFlow_ 2.0 and Keras_ are supported through graph freezing_.

ScikitLearn_ and Spark_ models are supported by SmartSim as well
through the use of the ONNX_ runtime.

.. _Spark: https://spark.apache.org/mllib/
.. _Keras: https://keras.io
.. _ScikitLearn: https://github.com/scikit-learn/scikit-learn
.. _TensorFlow: https://github.com/tensorflow/tensorflow
.. _PyTorch: https://github.com/pytorch/pytorch
.. _ONNX: https://github.com/microsoft/onnxruntime
.. _RedisAI: https://github.com/RedisAI/RedisAI
.. _freezing: https://github.com/leimao/Frozen-Graph-TensorFlow

------------------------------------------------------------

=============
Prerequisites
=============

The prerequisites to install SmartSim and SmartRedis are:

  - Python 3.7.x (or later) and pip
  - CMake 3.10.x (or later)
  - GCC > 5
  - GNU Make > 4.0
  - git-lfs

For most developer systems, many of these packages will already
be installed.

Git LFS can be installed through ``conda install git-lfs && git lfs install``

Be sure to reference the :ref:`installation notes <install-notes>` for helpful
information regarding various system types before installation.


========
SmartSim
========

Activate a new virtual environment and install SmartSim from PyPi with
the following command

.. code-block:: bash

    pip install smartsim

If you would like SmartSim to also install Machine Learning libraries that
can be used outside SmartSim to build SmartSim-compatible models, you
can request their installation through the ``ml`` flag as follows:

.. code-block:: bash

    pip install smartsim[ml]


At this point, SmartSim is installed and can be used for more basic features.
If you want to use the machine learning features of SmartSim, you will need
to install the ML backends in the section below.


Install ML Backends
===================

Use the ``smart`` cli tool to install the machine learning backends that
are built into the Orchestrator database.
``smart`` is installed during the pip installation of SmartSim and may
only be available while your virtual environment is active.

To see all the installation options:

.. code-block:: bash

    smart

To install the database ML backends for CPU, run

.. code-block:: bash

    # run one of the following
    smart --device cpu          # install PT and TF for cpu
    smart --device cpu --onnx   # install all backends (PT, TF, ONNX) on gpu

By default, ``smart`` will install PyTorch and TensorFlow backends
for use in SmartSim.

.. note::

    If a re-build is needed for any reason, ``smart --clean`` will remove
    all of the previous installs for the ML backends and ``smart --clobber`` will
    remove all pre-built dependencies as well as the ML backends.


To install the database ML backends for GPU, set the following environment variables if
CUDNN is not in your ``LD_LIBRARY_PATH`` or default loader locations.

  - ``CUDNN_INCLUDE_DIR``  - path to directory containing cudnn.h
  - ``CUDNN_LIBRARY``      - path to directory containing libcudnn.so

For example, for bash do

.. code-block:: bash

    export CUDNN_LIBRARY=/lus/sonexion/spartee/cuda/lib64/
    export CUDNN_INCLUDE_DIR=/lus/sonexion/spartee/cuda/include/
    export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH


.. code-block:: bash

    # run one of the following
    smart --device gpu          # install PT and TF for gpu
    smart --device gpu --onnx   # install all backends (PT, TF, ONNX) on gpu


.. note::

  Currently, SmartSim is solely compatible with NVIDIA GPUs on Linux systems
  and ``CUDA >= 11`` is required to build.

.. note::

  If the ``smart`` tool is not found. Look for it in places like
  ``~/.local/bin`` and other ``bin`` locations and add it to your
  ``$PATH``

----------------------------------------------------------------------

==========
SmartRedis
==========

There are implementations of the SmartRedis client in
4 languages: Python, C++, C and Fortran. The Python
client is installed through ``pip`` and the compiled
clients can be built as a static or shared library
through cmake.

SmartRedis Python supports the same architectures for
pre-built wheels that SmartSim does.

.. list-table:: Supported Systems for Pre-built Wheels
   :widths: 50 50
   :header-rows: 1
   :align: center

   * - Platform
     - Python Versions
   * - MacOS
     - 3.7 - 3.9
   * - Linux
     - 3.7 - 3.9


The Python client for SmartRedis is installed through
``pip`` as follows:

.. include:: ../smartredis/doc/install/python_client.rst


Build SmartRedis Library (C++, C, Fortran)
==========================================


.. include:: ../smartredis/doc/install/lib.rst


-----------------------------------------------------------------


===========
From Source
===========

This section will be geared towards contributors who want
to install SmartSim and SmartRedis from source. If you are
installing from source for other reasons, follow the steps
below but use the distribution provided hosted on GitHub
or PyPi.


Install SmartSim from Source
============================

First, clone SmartSim.

.. code-block:: bash

  git clone https://github.com/CrayLabs/SmartSim smartsim

And then install SmartSim with pip in *editable* mode. This way,
SmartSim is installed in your virtual environment and available
in PYTHONPATH, but the source remains at the site of the clone
instead of in site-packages.

.. code-block:: bash

  cd smartsim
  pip install -e .[dev,ml]   # for bash users
  pip install -e .\[dev,ml\] # for zsh users

Use the now installed ``smart`` cli to install the machine learning
runtimes.

.. code-block:: bash

  # run one of the following
  smart -v --device cpu          # verbose install cpu
  smart -v --device gpu          # verbose install gpu
  smart -v --device gpu --onnx   # install all backends (PT, TF, ONNX) on gpu


Install SmartRedis from Source
==============================

.. include:: ../smartredis/doc/install/from_source.rst


Building the Documentation
==========================

.. note::
    To build the full documentation, users need to install
    ``doxygen 1.9.1``. For Mac OS users, doxygen can be
    installed through ``brew install doxygen``

Users can optionally build documentation of SmartSim

.. code-block:: bash

  git clone https://github.com/CrayLabs/SmartRedis.git
  cd /smartsim      # top level smartsim dir
  make docs

Once the documentation has successfully built, users can open the
main documents page from ``doc/_build/html/index.html``


.. _install-notes:

============================================
Installation Notes for Specific System Types
============================================


The following describes installation details for
various system types that SmartSim may be used on.


SmartSim on MacOS
=================

We recommend users and contributors install brew_ for managing installed packages.
For contributors, the following brew packages can be helpful

 - openmpi_ for building and running parallel SmartRedis examples
 - doxygen_ for building the documention
 - cmake_ for building SmartSim and SmartRedis from source

.. _brew: https://brew.sh/
.. _openmpi: https://formulae.brew.sh/formula/open-mpi#default
.. _doxygen: https://formulae.brew.sh/formula/doxygen#default
.. _cmake: https://formulae.brew.sh/formula/cmake#default

For Mac OS users, the version of ``make`` that comes with
the Mac command line tools is often 3.81 which needs to be updated to install
SmartSim. Users can ``brew install make`` to get ``make`` > 4.0 but
brew will install it as ``gmake``. An easy way around this
is to do ``alias make=gmake``.


SmartSim on Ubuntu or Linux Workstations
========================================

When building SmartSim for Linux systems where the user
has root access, many of the needed packages can be installed
through ``apt``.

If you have a CUDA enabled GPU and want to run SmartSim on GPU
on a Linux system you have root access to, you can install CUDA
through ``apt`` with the ``cuda`` package.

In addition, cuDNN can be installed through the ``libcudnn``
and ``libcudnn-dev`` pacakges.

If you run into trouble compiling the machine learning runtimes
on Ubuntu because of cuDNN, it might be due to this issue_

.. _issue: https://github.com/pytorch/pytorch/issues/40965


SmartSim on Cray XC, CS, and EX
===============================

If on a Cray system, be sure to set the correct toolchain. SmartSim
is tested on ``PrgEnv-GNU`` and ``PrgEnv-Cray`` modules.

.. note::

    If on a Cray, please note that the intel and PGI compiler
    toolchains are currently not supported by SmartSim.

Before installing the machine learning runtimes with the ``smart``
cli tool, be sure to set the ``CRAYPE_LINK_TYPE`` to ``dynamic``

.. code-block:: bash

    export CRAYPE_LINK_TYPE=dynamic

Keep in mind, the libraries installed above need to be accessible
by SmartSim at runtime. If using a networked file system (NFS),
make sure to install these somewhere reachable from head, MOM, and
compute nodes (network mounted).

CUDA and CUDNN on Cray
----------------------

Usually ``cudatoolkit`` is available as a module and CUDA
is installed in ``/usr/local/cuda``. In this case, prior
to installation run

.. code-block:: bash

    module load cudatoolkit

If cuDNN libraries and includes are not installed as a module
or otherwise, you can install them with ``conda``.

.. code-block:: bash

  # Install CUDA requirements
  conda install cudatoolkit cudnn
  export CUDNN_LIBRARY=/path/to/miniconda/pkgs/cudnn-x.x.x-cudax.x_x/lib
  export CUDNN_INCLUDE_DIR=/path/to/miniconda/pkgs/cudnn-x.x.x-cudax.x_x/include

Be sure to get the cuDNN version that matches your CUDA installation
The package_ names usually specify the versions.

.. _package: https://anaconda.org/anaconda/cudnn/files


SmartSim on Summit at OLCF
==========================

Since SmartSim does not have a built PowerPC build, the build steps for
an IBM system are slightly different than other systems.

Luckily for us, Summit has an environment with many of the ML dependencies
that SmartSim needs already built into it. Users can follow these instructions
to get a working SmartSim build with PyTorch for GPU on Summit.

.. code-block:: bash

  # setup Python and build environment
  module load open-ce
  conda create -p /ccs/home/<USERNAME>/.conda/envs/smartsim --clone open-ce-1.2.0-py38-0
  conda activate smartsim
  module load gcc/8.1.1
  module load cuda/11.2.0
  module unload xalt
  export CC=$(which gcc)
  export CXX=$(which g++)
  export LDFLAGS="$LDFLAGS -pthread"
  export CUDNN_LIBRARY=/sw/summit/cuda/11.2.0/lib64
  export CUDNN_INCLUDE_DIR=/sw/summit/cuda/11.2.0/include/

  # clone SmartRedis and build
  git clone https://github.com/SmartRedis.git smartredis
  pushd smartredis
  make lib && pip install .
  popd

  # clone SmartSim and build
  git clone https://github.com/SmartSim.git smartsim
  pushd smartsim
  pip install .

  export Torch_DIR=/ccs/home/<USERNAME>/.conda/envs/smartsim/lib/python3.8/site-packages/torch/share/cmake/Torch/
  export CFLAGS="$CFLAGS -I/ccs/home/<USERNAME>/.conda/envs/smarter/lib/python3.8/site-packages/tensorflow/include"
  # install PyTorch backend for the Orchestrator database.
  smart --device=gpu --torch_dir $Torch_DIR -v



SmartSim on Cheyenne at NCAR
============================

Since SmartSim does not currently support the Message Passing Toolkit (MPT), Cheyenne
users of SmartSim will need to utilize OpenMPI.

The following module commands were utilized to run the examples

.. code-block:: bash

  $ module purge
  $ module load ncarenv/1.3 gnu/8.3.0 ncarcompilers/0.5.0 netcdf/4.7.4 openmpi/4.0.5

With this environment loaded, users will need to build and install both SmartSim and
SmartRedis through pip. Usually we recommend users installing or loading miniconda and
using the pip that comes with that installation.

.. code-block:: bash

  $ pip install smartsim
  $ smart --device cpu  (Since Cheyenne does not have GPUs)

To make the SmartRedis library (C, C++, Fortran clients), follow these steps with
the same environment loaded.

.. code-block:: bash

  # clone SmartRedis and build
  $ git clone https://github.com/SmartRedis.git smartredis
  $ cd smartredis
  $ make lib


==============================================
Changing Redis installation and Configurations
==============================================

If you want to use a pre-existing Redis installation, override
the configuration, or install the dependencies for SmartSim
manually, you can do so by specifying a SmartSim configuration
file in a ``config.toml``

This can be useful for
 - contributors who want to try bleeding edge features
 - users who want to edit Redis behavior (adding checkpoints)
 - sites who want to package SmartSim in a different manner.
 - sites who want to set intervals for scheduler communication

Override Pre-packaged Settings
==============================

The following fields can be overridden in the SmartSim configuration
file

 - Redis installation location
 - Redis configuration file location
 - RedisAI module installation location
 - Test launcher/device/network interface
 - Log level default
 - Job manager interval default

.. note::

    If you want to override the configuration, you need
    to supply values for each of these.

Install the SmartSim user configuration file.
Usually this is in your ``$HOME`` directory under
``~/.smartsim/config.toml``, but for users of networked filesystems
where the ``$HOME`` directory is not mounted, do the following

.. code-block:: bash

    # set the configuration directory to somewhere accessible
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
  # number of seconds per job status update
  # for jobs on WLM system (e.g. slurm, pbs, etc)
  jm_interval = 15    # default
  log_level = "info" # default

  [redis]
  # path to where "redis-server" and "redis-cli" binaries are located
  bin = "/path/to/redis/src/"
  config = "/path/to/redis.conf" # optional!

    [redis.modules]
    ai = "/path/to/RedisAI/install-cpu/"

  [test]
  launcher = "local" # default
  interface = "ib0" # network interface to use in tests
  device = "CPU" # device to use for tests (CPU/GPU)
