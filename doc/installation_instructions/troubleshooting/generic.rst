Customizing environment variables
---------------------------------

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

Compiler environment
^^^^^^^^^^^^^^^^^^^^

Unlike SmartRedis, we *strongly* encourage users to only use the GNU compiler
chain to build the SmartSim dependencies. Notably, RedisAI has some coding
conventions that prevent the use of Intel compiler chain. If a specific
compiler should be used (e.g. the Cray Programming Environment wrappers),
the following environment variables will control the C and C++ compilers:

- ``CC``: Path to the C compiler
- ``CXX``: Path the C++ compiler