NERSC Perlmutter
================

One-time Setup
--------------

To install SmartSim on Perlmutter, follow these steps:

**Step 1:** Create and activate a conda environment for SmartSim:

.. code:: bash

    module load conda
    conda create -n smartsim python=3.11
    conda activate smartsim

**Step 2:** Install SmartSim in the conda environment:

.. code:: bash

    pip install git+https://github.com/CrayLabs/SmartSim.git

**Step 3:** Build Redis, RedisAI, the backends, and all the Python packages:

.. code:: bash

    module load cudatoolkit/12.2 cudnn/8.9.3_cuda12
    smart build --device=cuda-12

**Step 4:** Check that SmartSim has been installed and built correctly:

.. code:: bash

    smart validate --device gpu

The following output indicates a successful install:

.. code:: bash

    [SmartSim] INFO Verifying Tensor Transfer
    [SmartSim] INFO Verifying Torch Backend
    [SmartSim] INFO Verifying ONNX Backend
    [SmartSim] INFO Verifying TensorFlow Backend
    16:26:35 login SmartSim[557020:MainThread] INFO Success!

Post-installation
-----------------

After completing the above steps to install SmartSim in a conda environment, you
can reload the conda environment by running the following commands:

.. code:: bash

    module load conda cudatoolkit/12.2 cudnn/8.9.3_cuda12
    conda activate smartsim
