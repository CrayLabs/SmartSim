NERSC Perlmutter
================

One-time Setup
--------------

To install SmartSim on Perlmutter please follow the following instructions.

First create and activate a conda environment for SmartSim

.. code:: bash

    module load conda
    conda create -n smartsim python=3.11
    conda activate smartsim

Next we will install SmartSim in this environment

.. code:: bash

    pip install git+https://github.com/ashao/SmartSim.git@refactor_rai_builder

Next build Redis, RedisAI, the backends, and all the python packages

.. code:: bash

    module load cudatoolkit/12.2 cudnn/8.9.3_cuda12
    smart build --device=cuda-12

To double check that SmartSim has been installed and built correctly, run

.. code:: bash

    smart validate --device gpu

Post-installation
-----------------

After following the above steps when first installing SmartSim, subsequently
you can use load the SmartSim environment using

.. code:: bash

    module load conda cudatoolkit/12.2 cudnn/8.9.3_cuda12
    conda activate smartsim
