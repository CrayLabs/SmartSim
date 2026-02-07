PML Scylla
==========

.. warning::
    As of September 2024, the software stack on Scylla is still being finalized.
    Therefore, please consider these instructions as preliminary for now.

One-time Setup
--------------

To install SmartSim on Scylla, follow these steps:

**Step 1:** Create and activate a Python virtual environment for SmartSim:

.. code:: bash

    module use module use /scyllapfs/hpe/ashao/smartsim_dependencies/modulefiles
    module load cudatoolkit cudnn git
    python -m venv /scyllafps/scratch/$USER/venvs/smartsim
    source /scyllafps/scratch/$USER/venvs/smartsim/bin/activate

**Step 2:** Build the SmartRedis C++ and Fortran libraries:

.. code:: bash

    git clone https://github.com/CrayLabs/SmartRedis.git
    cd SmartRedis
    make lib-with-fortran
    pip install .
    cd ..

**Step 3:** Install SmartSim in the conda environment:

.. code:: bash

    pip install git+https://github.com/CrayLabs/SmartSim.git

**Step 4:** Build Redis, RedisAI, the backends, and all the Python packages:

.. code:: bash

    export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" # Workaround for a PyTorch problem
    smart build --device=cuda-12
    module unload cudnn # Workaround for a PyTorch problem


.. note::
    The first workaround is needed because for some reason the autodetection
    of CUDA architectures is not consistent internally with one of PyTorch's
    dependencies. This seems to be unique to this machine as we do not see
    this on other platforms.

    The second workaround is needed because PyTorch 2.3 (and possibly 2.2)
    will attempt to load the version of cuDNN that is in the LD_LIBRARY_PATH
    instead of the version shipped with PyTorch itself. This results in
    unfound symbols.

**Step 5:** Check that SmartSim has been installed and built correctly:

.. code:: bash

    srun -n 1 -p gpu --gpus=1 --pty smart validate --device gpu

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

    module load cudatoolkit/12.4.1 git # cudnn should NOT be loaded
    source /scyllafps/scratch/$USER/venvs/smartsim/bin/activate

