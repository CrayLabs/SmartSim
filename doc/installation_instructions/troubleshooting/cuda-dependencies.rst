Nvidia GPU Dependencies
-----------------------

The Nvidia installation instructions for CUDA Toolkit and cuDNN tend to be
tailored for users with root access. For those on HPC platforms where root
access is rare, users can install Nvidia dependencies in user-space.  Even on
machines where these dependencies are available, if environment variables are
not set, the ``smart build`` step may fail. This section details how to download
and install these dependencies and configure your build environment.

.. note::

    At runtime, the environment in which the Orchestrator is launched must have
    the cuDNN and CUDA Toolkit libraries findable by the link loader (e.g.
    available in the ``LD_LIBRARY_PATH`` environment variable).

Download and install
^^^^^^^^^^^^^^^^^^^^

**Step 1:** Find a location which is globally accessible and has sufficient
storage space (about 12GB) and set an environment variable

.. code-block:: bash

    export CUDA_TOOLKIT_INSTALL_PATH=/path/to/install/location/cudatoolkit
    export CUDNN_INSTALL_PATH=/path/to/install/location/cudnn

**Step 2:** Download cudatoolkit and install it

.. tabs::

    .. group-tab:: CUDA 11

        .. code-block:: bash

            wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
            sh ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent --toolkitpath=$CUDA_TOOLKIT_INSTALL_PATH

    .. group-tab:: CUDA 12

        .. code-block:: bash

            wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run
            sh ./cuda_12.5.0_555.42.02_linux.run --toolkit --silent --toolkitpath=$CUDA_TOOLKIT_INSTALL_PATH

**Step 3:** Download cuDNN
For cuDNN, follow `Nvidia's instructions
<https://docs.nvidia.com/deeplearning/cudnn/installation/overview.html>`_ for
downloading cuDNN version 8.9 for either CUDA-11 or CUDA-12.

**Step 4:** Untar the cuDNN archive

.. tabs::

    .. group-tab:: CUDA 11

        .. code-block:: bash

            mkdir -p $CUDNN_INSTALL_PATH
            tar -xf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar -C $CUDNN_INSTALL_PATH --strip-components 1

    .. group-tab:: CUDA 12

        .. code-block:: bash

            mkdir -p $CUDNN_INSTALL_PATH
            tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar -C $CUDNN_INSTALL_PATH --strip-components 1

Option 1: Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following environment variables help the ``smart build`` step find and link in the
CUDA Toolkit and cuDNN libraries needed to build the ML backends.

.. code-block:: bash

    # CUDA Toolkit variables
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_INSTALL_PATH
    export CUDA_NVCC_EXECUTABLE=$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc
    export CUDA_INCLUDE_DIRS=$CUDA_TOOLKIT_ROOT_DIR/include

    # cuDNN Variables
    export CUDNN_LIBRARY=$CUDNN_INSTALL_PATH/lib/libcudnn.so
    export CUDNN_INCLUDE_DIR=$CUDNN_INSTALL_PATH/include

Option 2: Setup Modulefiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, these environment variables can be setup by using environment
modules instead. This can be especially useful when the CUDA dependencies are
intended to be shared across users.

**Step 1:** Download these two modulefiles to a directory of your choosing

- :download:`CUDA Toolkit <./cudatoolkit>`
- :download:`cuDNN <./cudnn>`

**Step 2:** Modify the files to set the ``cuda_home`` and ``CUDNN_ROOT``
variables to match the installed locations for CUDA Toolkit and cuDNN.

**Step 3:** In your ``.bashrc`` add the following line

.. code-block::

    module use /path/to/modulefile root

**Step 4:** Activate the modulefiles

.. code-block::

    module load cudatoolkit cudnn