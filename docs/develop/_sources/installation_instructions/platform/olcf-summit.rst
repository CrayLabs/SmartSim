
Summit at OLCF
==============

Since SmartSim does not have a built PowerPC build, the build steps for an IBM
system are slightly different than other systems.

Luckily for us, a conda channel with all relevant packages is maintained as part
of the `OpenCE <https://github.com/open-ce/open-ce>`_
initiative.  Users can follow these instructions to get a working SmartSim build
with PyTorch and TensorFlow for GPU on Summit.  Note that SmartSim and SmartRedis
will be downloaded to the working directory from which these instructions are executed.

Note that the available PyTorch version (1.10.2) does not match
the one expected by RedisAI 1.2.7 (1.11): it is still compatible and should
work, but please open an issue on SmartSim's GitHub repo if you run
into problems.

.. code-block:: bash

  # setup Python and build environment
  export ENV_NAME=smartsim-0.8.0
  git clone https://github.com/CrayLabs/SmartRedis.git smartredis
  git clone https://github.com/CrayLabs/SmartSim.git smartsim
  conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/1.6.1/
  conda create --name $ENV_NAME -y  python=3.9 \
                                    git-lfs \
                                    cmake \
                                    make \
                                    cudnn=8.1.1_11.2 \
                                    cudatoolkit=11.2.2 \
                                    tensorflow=2.8.1 \
                                    libtensorflow \
                                    pytorch=1.10.2 \
                                    torchvision=0.11.3
  conda activate $ENV_NAME
  export CC=$(which gcc)
  export CXX=$(which g++)
  export LDFLAGS="$LDFLAGS -pthread"
  export CUDNN_LIBRARY=/ccs/home/$USER/.conda/envs/$ENV_NAME/lib/
  export CUDNN_INCLUDE_DIR=/ccs/home/$USER/.conda/envs/$ENV_NAME/include/
  module load cuda/11.4.2
  export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH:/ccs/home/$USER/.conda/envs/$ENV_NAME/lib/python3.9/site-packages/torch/lib
  module load gcc/9.3.0
  module unload xalt
  # clone SmartRedis and build
  pushd smartredis
  make lib && pip install .
  popd

  # clone SmartSim and build
  pushd smartsim
  pip install .

  # install PyTorch and TensorFlow backend for the Orchestrator database.
  export Torch_DIR=/ccs/home/$USER/.conda/envs/$ENV_NAME/lib/python3.9/site-packages/torch/share/cmake/Torch/
  export CFLAGS="$CFLAGS -I/ccs/home/$USER/.conda/envs/$ENV_NAME/lib/python3.9/site-packages/tensorflow/include"
  export SMARTSIM_REDISAI=1.2.7
  export Tensorflow_BUILD_DIR=/ccs/home/$USER/.conda/envs/$ENV_NAME/lib/python3.9/site-packages/tensorflow/
  smart build --device=gpu --torch_dir $Torch_DIR --libtensorflow_dir $Tensorflow_BUILD_DIR -v

  # Show LD_LIBRARY_PATH for future reference
  echo "SmartSim installation is complete, LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

When executing SmartSim, if you want to use the PyTorch and TensorFlow backends
in the orchestrator, you will need to set up the same environment used at build
time:

.. code-block:: bash

  module load cuda/11.4.2
  export CUDNN_LIBRARY=/ccs/home/$USER/.conda/envs/$ENV_NAME/lib/
  export LD_LIBRARY_PATH=/ccs/home/$USER/.conda/envs/smartsim/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH:$CUDNN_LIBRARY
  module load gcc/9.3.0
  module unload xalt
