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

The base prerequisites to install SmartSim and SmartRedis are:

  - Python 3.7-3.9
  - Pip
  - Cmake 3.13.x (or later)
  - C compiler
  - C++ compiler
  - GNU Make > 4.0
  - git
  - `git-lfs`_

.. _git-lfs: https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing

.. note::

  GCC 5-9, 11, and 12 is recommended. There are known bugs with GCC 10.

.. _GPU Support:

GPU Support
===========

The machine-learning backends have additional requirements in order to
use GPUs for inference

  - `CUDA Toolkit 11 (11.4 recommended) <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_
  - `cuDNN 8 (tested with 8.2.1 and 8.4.0) <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download>`_
  - OS: Linux
  - GPU: Nvidia

Be sure to reference the :ref:`installation notes <install-notes>` for helpful
information regarding various system types before installation.

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
     - Not supported
     - 3.7 - 3.9
   * - Linux
     - x86_64
     - Nvidia
     - 3.7 - 3.9


.. note::

    Windows is not supported and there are currently no plans
    to support Windows.


Native support for various machine learning libraries and their
versions is dictated by our dependency on RedisAI_ 1.2.7. Users
can also select RedisAI 1.2.3 or 1.2.5 (though that also limits
the version of the ML libraries).

+------------------+----------+-------------+---------------+
| RedisAI          | PyTorch  | Tensorflow  | ONNX Runtime  |
+==================+==========+=============+===============+
| 1.2.7 (default)  | 1.11.0   | 2.8.0       | 1.11.1        |
+------------------+----------+-------------+---------------+
| 1.2.5            | 1.9.0    | 2.6.0       | 1.9.0         |
+------------------+----------+-------------+---------------+
| 1.2.3            | 1.7.0    | 2.5.2       | 1.9.0         |
+------------------+----------+-------------+---------------+

TensorFlow_ 2.0 and Keras_ are supported through `graph freezing`_.

ScikitLearn_ and Spark_ models are supported by SmartSim as well
through the use of the ONNX_ runtime (which is not built by
default due to issues with glibc on a variety of Linux
platforms and lack of support for Mac OS X).

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
==========

We recommend users and contributors install brew_ for managing installed
packages.  For contributors, the following brew packages can be helpful:

- openmpi_ for building and running parallel SmartRedis examples
- doxygen_ for building the documention
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

We first recommend creating a new virtual environment:

.. code-block:: bash

    python -m venv /path/to/new/environment
    source /path/to/new/environment/bin/activate

and install SmartSim from PyPI with the following command:

.. code-block:: bash

    pip install smartsim

If you would like SmartSim to also install Machine Learning libraries that
can be used outside SmartSim to build SmartSim-compatible models, you
can request their installation through the ``ml`` flag as follows:

.. code-block:: bash

    # For bash
    pip install smartsim[ml]
    # or if using ZSH
    pip install smartsim\[ml\]

At this point, SmartSim is installed and can be used for more basic features.
If you want to use the machine learning features of SmartSim, you will need
to install the ML backends in the section below.


Step 2: Build SmartSim
======================

Use the ``smart`` cli tool to install the machine learning backends that
are built into the Orchestrator database. ``smart`` is installed during
the pip installation of SmartSim and may only be available while your
virtual environment is active.

To see all the installation options:

.. code-block:: bash

    smart

CPU Install
-----------

To install the default ML backends for CPU, run

.. code-block:: bash

    # run one of the following
    smart build --device cpu          # install PT and TF for cpu
    smart build --device cpu --onnx   # install all backends (PT, TF, ONNX) on gpu

By default, ``smart`` will install PyTorch and TensorFlow backends
for use in SmartSim.

.. note::

    If a re-build is needed for any reason, ``smart clean`` will remove
    all of the previous installs for the ML backends and ``smart clobber`` will
    remove all pre-built dependencies as well as the ML backends.


GPU Install
-----------

With the proper environment setup (see :ref:`GPU support`) the only difference
to building SmartSim with GPU support is to specify a different ``device``

.. code-block:: bash

    # run one of the following
    smart build --device gpu          # install PT and TF for gpu
    smart build --device gpu --onnx   # install all backends (PT, TF, ONNX) on gpu

.. note::

  GPU builds can be troublesome due to the way that RedisAI and the ML-package
  backends look for the CUDA Toolkit and cuDNN libraries. Please see the
  :ref:`Platform Installation Section <install-notes>` section for guidance.

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
     - 3.7 - 3.9
   * - Linux
     - 3.7 - 3.9

The Python client for SmartRedis is installed through ``pip`` as follows:

.. include:: ../../smartredis/doc/install/python_client.rst

Build SmartRedis Library (C++, C, Fortran)
==========================================

.. include:: ../../smartredis/doc/install/lib.rst

===========
From Source
===========

This section will be geared towards contributors who want to install SmartSim
and SmartRedis from source. If you are installing from source for other reasons,
follow the steps below but use the distribution provided hosted on GitHub or
PyPI.

.. _from-source:

Install SmartSim from Source
============================

First, clone SmartSim.

.. code-block:: bash

  git clone https://github.com/CrayLabs/SmartSim smartsim

And then install SmartSim with pip in *editable* mode. This way, SmartSim is
installed in your virtual environment and available in PYTHONPATH, but the
source remains at the site of the clone instead of in site-packages.

.. code-block:: bash

  cd smartsim
  pip install -e .[dev,ml]   # for bash users
  pip install -e .\[dev,ml\] # for zsh users

Use the now installed ``smart`` cli to install the machine learning runtimes.

.. code-block:: bash

  # run one of the following
  smart build -v --device cpu          # verbose install cpu
  smart build -v --device gpu          # verbose install gpu
  smart build -v --device gpu --onnx   # install all backends (PT, TF, ONNX) on gpu


Build the SmartRedis library
============================

.. include:: ../../smartredis/doc/install/lib.rst



