GPU dependencies (non-root)
===========================

The Nvidia installation instructions for CUDA Toolkit and cuDNN tend to be
tailored for users with root access. For those on HPC platforms where root
access is rare, manually downloading and installing these dependencies as
a user is possible.

.. code-block:: bash

    wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
    chmod +x cuda_11.4.4_470.82.01_linux.run
    ./cuda_11.4.4_470.82.01_linux.run --toolkit  --silent --toolkitpath=/path/to/install/location/

For cuDNN, follow `Nvidia's instructions
<https://docs.nvidia.com/deeplearning/cudnn/installation/overview.html>`_,
and copy the cuDNN libraries to the `lib64` directory at the CUDA Toolkit
location specified above.