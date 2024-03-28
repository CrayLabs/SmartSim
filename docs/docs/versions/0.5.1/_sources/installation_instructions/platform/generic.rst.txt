Customizing environment variables
=================================

Various environment variables can be used to control the compilers and
dependencies for SmartSim. These are particularly important to set before the
``smart build`` step to ensure that the Orchestrator and machine-learning
backends are compiled with the desired compilation environment.

.. note::

    The compilation environment that SmartSim is compiled with *does not*
    necessarily have to be compatible with the SmartRedis library and the
    simulation application that will be launched by SmartSim. To ensure
    that this works as intended however, please be sure to set the
    correct environment for the simulation using the ``RunSettings``.

All of the following environment variables must be *exported* to ensure that
they are used throughout the entire build process. Additionally at runtime, the
environment in which the Orchestrator is launched must have the cuDNN and CUDA
Toolkit libraries findable by the link loader (e.g. available in the
``LD_LIBRARY_PATH`` environment variable).

Compiler environment
--------------------

Unlike SmartRedis, we *strongly* encourage users to only use the GNU compiler
chain to build the SmartSim dependencies. Notably, RedisAI has some coding
conventions that prevent the use of Intel compiler chain. If a specific
compiler should be used (e.g. the Cray Programming Environment wrappers),
the following environment variables will control the C and C++ compilers:

- ``CC``: Path to the C compiler
- ``CXX``: Path the C++ compiler

CUDA-related
------------

The following environment variables help the ``smart build`` step find and link in the
CUDA Toolkit and cuDNN libraries needed to build the ML backends.

cuDNN:

- ``CUDNN_LIBRARY``: Path to the cuDNN shared libraries (e.g. ``libcudnn.so``) are found
- ``CUDNN_INCLUDE_DIR``: Path to cuDNN header files (e.g. ``cudnn.h``)

CUDA Toolkit:

- ``CUDA_TOOLKIT_ROOT_DIR``: Path to the root directory of CUDA Toolkit
- ``CUDA_NVCC_EXECUTABLE``: Path to the ``nvcc`` compiler
- ``CUDA_INCLUDE_DIRS``: Path to the CUDA Toolkit headers

