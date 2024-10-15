.. _basic_install_SS:

******************
Basic Installation
******************

The following will show how to install both SmartSim and SmartRedis.

.. note::

  For users on platforms with a 'site install' of SmartSim please follow
  :ref:`Site Installation <site_installation>`.

=============
Prerequisites
=============

Basic
=====

The base prerequisites to install SmartSim and SmartRedis wtih CPU-only support are:

  - Python 3.9-3.11
  - Pip
  - Cmake 3.13.x (or later)
  - C compiler
  - C++ compiler
  - GNU Make > 4.0
  - git

.. note::

  GCC 9, 11-13 is recommended (here are known issues compiling with GCC 10). For
  CUDA 11.8, GCC 9 or 11 must be used.

.. warning::

  Apple Clang 15 seems to have issues on MacOS with Apple Silicon. Please modify
  your path to ensure that a version of GCC installed by brew has priority. Note
  this seems to be hardcoded to `gcc` and `g++` in the Redis build so ensure that
  `which gcc g++` do not point to Apple Clang.


ML Library Support
==================

We currently support both Nvidia and AMD GPUs when using RedisAI for GPU inference. The support
for these GPUs often depends on the version of the CUDA or ROCm stack that is availble on your
machine. In _most_ cases, the versions backwards compatible. If you encounter problems, please
contact us and we can build the backend libraries for your desired version of CUDA and ROCm.

CPU backends are provided for Apple (both Intel and Apple Silicon) and Linux (x86_64).

Be sure to reference the table below to find which versions of the ML libraries are supported for
your particular platform. Additional, see :ref:`installation notes <install-notes>` for helpful
information regarding various system types before installation.

Linux
-----

.. tabs::

    .. group-tab:: CUDA 11

      Additional requirements:

      * GCC <= 11
      * CUDA Toolkit 11.7 or 11.8
      * cuDNN 8.9

      .. list-table:: Nvidia CUDA 11
         :widths: 50 50 50 50
         :header-rows: 1
         :align: center

         * - Python Versions
           - Torch
           - Tensorflow
           - ONNX Runtime
         * - 3.9-3.11
           - 2.3.1
           - 2.14.1
           - 1.17.3

    .. group-tab:: CUDA 12

      Additional requirements:

      * CUDA Toolkit 12
      * cuDNN 8.9

      .. list-table:: Nvidia CUDA 12
         :widths: 50 50 50 50
         :header-rows: 1
         :align: center

         * - Python Versions
           - Torch
           - Tensorflow
           - ONNX Runtime
         * - 3.9-3.11
           - 2.3.1
           - 2.17
           - 1.17.3

    .. group-tab:: ROCm 6

      .. list-table:: AMD ROCm 6.1
         :widths: 50 50 50 50
         :header-rows: 1
         :align: center

         * - Python Versions
           - Torch
           - Tensorflow
           - ONNX Runtime
         * - 3.9-3.11
           - 2.4.1
           - N/A
           - N/A

    .. group-tab:: CPU

      .. list-table:: CPU-only
         :widths: 50 50 50 50
         :header-rows: 1
         :align: center

         * - Python Versions
           - Torch
           - Tensorflow
           - ONNX Runtime
         * - 3.9-3.11
           - 2.4.0
           - 2.15
           - 1.17.3

MacOSX
------

.. tabs::

    .. group-tab:: Apple Silicon

      .. list-table:: Apple Silicon ARM64 (no Metal support)
         :widths: 50 50 50 50
         :header-rows: 1
         :align: center

         * - Python Versions
           - Torch
           - Tensorflow
           - ONNX Runtime
         * - 3.9-3.11
           - 2.4.0
           - 2.17
           - 1.17.3

    .. group-tab:: Intel Mac (x86)

      .. list-table:: CPU-only
         :widths: 50 50 50 50
         :header-rows: 1
         :align: center

         * - Python Versions
           - Torch
           - Tensorflow
           - ONNX Runtime
         * - 3.9-3.11
           - 2.2.0
           - 2.15
           - 1.17.3


.. note::

    Users have succesfully run SmartSim on Windows using Windows Subsystem for Linux
    with Nvidia support. Generally, users should follow the Linux instructions here,
    however we make no guarantee or offer of support.


TensorFlow_ and Keras_ are supported through `graph freezing`_.

ScikitLearn_ and Spark_ models are supported by SmartSim as well
through the use of the ONNX_ runtime (which is not built by
default due to issues with glibc on a variety of Linux platforms).

.. _Spark: https://spark.apache.org/mllib/
.. _Keras: https://keras.io
.. _ScikitLearn: https://github.com/scikit-learn/scikit-learn
.. _TensorFlow: https://github.com/tensorflow/tensorflow
.. _PyTorch: https://github.com/pytorch/pytorch
.. _ONNX: https://github.com/microsoft/onnxruntime
.. _RedisAI: https://github.com/RedisAI/RedisAI
.. _graph freezing: https://github.com/leimao/Frozen-Graph-TensorFlow

------------------------------------------------------------

MacOS-only
============

We recommend users and contributors install brew_ for managing installed
packages. For contributors, the following brew packages can be helpful:

- openmpi_ for building and running parallel SmartRedis examples
- doxygen_ for building the documentation
- cmake_ for building SmartSim and SmartRedis from source

.. _brew: https://brew.sh/
.. _openmpi: https://formulae.brew.sh/formula/open-mpi#default
.. _doxygen: https://formulae.brew.sh/formula/doxygen#default
.. _cmake: https://formulae.brew.sh/formula/cmake#default

For Mac OS users, the version of ``make`` that comes with the Mac command line
tools is often 3.81 which needs to be updated to install SmartSim. Users can run
``brew install make`` to get ``make`` > 4.0 but brew will install it as
``gmake``. An easy way around this is to run ``alias make=gmake``.

.. _from-pypi:

========
SmartSim
========

There are two stages for the installation of SmartSim.

 1. `pip` install SmartSim Python package
 2. Build SmartSim using the `smart` command line tool

Step 1: Install Python Package
==============================

We first recommend creating a new
`virtual environment <https://docs.python.org/3/library/venv.html>`_:

.. code-block:: bash

    python -m venv /path/to/new/environment
    source /path/to/new/environment/bin/activate

and install SmartSim from PyPI with the following command:

.. code-block:: bash

    pip install smartsim

At this point, SmartSim can be used for describing and launching experiments, but
without any database/feature store functionality which allows for ML-enabled workflows.


Step 2: Build SmartSim
======================

Use the ``smart`` cli tool to install the machine learning backends that
are built into the Orchestrator database. ``smart`` is installed during
the pip installation of SmartSim and may only be available while your
virtual environment is active.

To see all the installation options:

.. code-block:: bash

    smart --help

.. code-block:: bash

    # run one of the following
    smart build --device cpu      # For unaccelerated AI/ML loads
    smart build --device cuda118  # Nvidia Accelerator with CUDA 11.8
    smart build --device cuda125  # Nvidia Accelerator with CUDA 12.5
    smart build --device rocm57   # AMD Accelerator with ROCm 5.7.0

By default, ``smart`` will install all backends available for the specified accelerator
_and_ the compatible versions of the Python packages associated with the backends. To
disable support for a specific backend, ``smart build`` accepts the flags
``--skip-torch``, ``--skip-tensorflow``, ``--skip-onnx`` which can also be used in
combination.

.. note::

    If a re-build is needed for any reason, ``smart clean`` will remove
    all of the previous installs for the ML backends and ``smart clobber`` will
    remove all pre-built dependencies as well as the ML backends.

.. note::

  GPU builds can be troublesome due to the way that RedisAI and the ML-package
  backends look for the CUDA Toolkit and cuDNN libraries. Please see the
  :ref:`Platform Installation Section <install-notes>` section for guidance.


.. _dragon_install:

Dragon Install
--------------

`Dragon <https://dragonhpc.github.io/dragon/doc/_build/html/index.html>`_ is
an HPC-native library for distributed computing. SmartSim can use Dragon as a
launcher on systems with Slurm or PBS as schedulers. To install the correct
version of Dragon, you can add the ``--dragon`` option to ``smart build``.
For example, to install dragon alongside the RedisAI CPU backends, you can run

.. code-block:: bash

    smart build --device cpu --dragon           # install Dragon, PT and TF for cpu

``smart build`` supports installing a specific version of dragon. It exposes the
parameters ``--dragon-repo`` and ``--dragon-version``, which can be used alone or
in combination to customize the Dragon installation. For example:

.. code-block:: bash

    # using the --dragon-repo and --dragon-version flags to customize the Dragon installation
    smart build --device cpu --dragon-repo userfork/dragon  # install Dragon from a specific repo
    smart build --device cpu --dragon-version 0.10          # install a specific Dragon release

    # combining both flags
    smart build --device cpu --dragon-repo userfork/dragon --dragon-version 0.91


.. note::
  Dragon is only supported on Linux systems. For further information, you
  can read :ref:`the dedicated documentation page <dragon>`.

==========
SmartRedis
==========

There are implementations of the SmartRedis client in 4 languages: Python, C++,
C, and Fortran. The Python client is installed through ``pip`` and the compiled
clients can be built as a static or shared library through ``cmake``.

SmartRedis Python supports the same architectures for pre-built wheels that
SmartSim does.

.. list-table:: Supported Systems for Pre-built Wheels
   :widths: 50 50
   :header-rows: 1
   :align: center

   * - Platform
     - Python Versions
   * - MacOS
     - 3.9 - 3.11
   * - Linux
     - 3.9 - 3.11

The Python client for SmartRedis is installed through ``pip`` as follows:

.. include:: ../../smartredis/doc/install/python_client.rst

Build SmartRedis Library (C++, C, Fortran)
==========================================

.. include:: ../../smartredis/doc/install/lib.rst

===========
From Source
===========

This section will be geared towards contributors who want to install SmartSim
and SmartRedis from source for development purposes. If you are installing
from source for other reasons, follow the steps below but use the source
distributions provided on GitHub or PyPI.

.. _from-source:

Install SmartSim from Source
============================

First, clone SmartSim.

.. code-block:: bash

  git clone https://github.com/CrayLabs/SmartSim smartsim


And then install SmartSim with pip in *editable* mode. This way, SmartSim is
installed in your virtual environment and available on `sys.path`, but the
source remains at the site of the clone instead of in site-packages.

.. code-block:: bash

  cd smartsim
  pip install -e .[dev]       # for bash users
  pip install -e ".[dev]"  # for zsh users

Use the now installed ``smart`` cli to install the machine learning runtimes and
dragon. Referring to "Step 2: Build SmartSim above".

Build the SmartRedis library
============================

.. include:: ../../smartredis/doc/install/lib.rst
