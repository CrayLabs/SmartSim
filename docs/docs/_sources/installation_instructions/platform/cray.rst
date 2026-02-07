HPE Cray supercomputers
=======================

On certain HPE Cray machines, the SmartSim dependencies have been installed
system-wide though specific paths and names might vary (please contact the team
if these instructions do not work).

.. code-block:: bash

    module use -a /lus/scratch/smartsim/local/modulefiles
    module load cudatoolkit/11.8 cudnn git-lfs

    module unload PrgEnv-cray PrgEnv-intel PrgEnv-gcc
    module load PrgEnv-gnu
    module switch gcc/11.2.0

    export CRAYPE_LINK_TYPE=dynamic

This should provide all the dependencies needed to build the GPU backends for
the ML bakcends. Users can thus proceed with their preferred way of installing
SmartSim either :ref:`from PyPI <from-pypi>` or :ref:`from source
<from-source>`.

