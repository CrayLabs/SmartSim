# Smart-Sim Library

    A library of tools dedicated to accelerating the convergence of AI and numerical
    simulation models. SmartSim can connect models written in Fortran, C, C++ and
    Python to the modern data science stack. Integration with workload managers like
    Slurm make it easy to run multiple jobs for simulation, analysis, and visualization
    all within a single allocation. Generate configurations and run ensembles of
    simulations all within the comfort of a jupyter notebook.

## Current Features

   - Clients in Python, C, C++ and Fortran (SILC)
   - Allocation management interface through Slurm
   - Ensembling through text-based configuration generation for models
   - Works on compute nodes for rapid prototyping and preprocessing
   - Runs inside Jupyter lab/notebook
   - Distributed, in-memory database
   - Pytorch, Tensorflow, and ONNX based inference suppport with RedisAI

## Build SmartSim

  - The following will describe how to build SmartSim in multiple settings.
  - General requirements
      - Python >= 3.7
      - GCC >= 5.0.0
      - Cmake >= 3.0.0
  - Tested on OSX, and many flavors of linux
  - Windows not supported.

### Get the Source

   - Clone the git repository
      - git clone {insert git address} Smart-Sim

### Get Python and Requirements

   - Install conda - https://docs.conda.io/en/latest/miniconda.html
   - Create a new conda environment
      - conda create --name py38 python=3.8
   - Activate the conda environment
      - conda activate py38
   - Install Dependencies
      - pip install -r requirements.txt
      - pip install -r requirements-dev.txt (for all requirements)

### GPU Install Requirements

   If you plan to run SmartSim's architecture on GPU you will need
   to follow the steps below. Otherwise if you plan to run solely
   on CPU, feel free to skip these steps

   For this portion you will want CUDA to be installed. Most likely,
   CUDA will be installed somewhere like `/usr/local/cuda`

   - Install CUDA requirements
      - conda install cudatoolkit=10.2 cudnn=7.6.5
   - (optional) Load CUDA module instead of using conda
      - module load cudatoolkit
   - Set CUDA environment variables
      - export CUDNN_LIBRARY=/path/to/miniconda3/pkgs/cudnn-7.6.5-cuda10.2_0/lib
      - export CUDNN_INCLUDE_DIR=/path/to/miniconda3/pkgs/cudnn-7.6.5-cuda10.2_0/include
      - export CUDATOOLKIT_HOME=/path/to/miniconda3/pkgs/cudatoolkit-10.2.89-hfd86e86_1/

### Building SmartSim with Dependencies

   There are multiple backends that can be used in the Redis-based
   orchestrator for online inference: Pytorch, TensorFlow, TF-Lite,
   and ONNX. Follow any *one* of these steps to build for the backend
   you are looking for.

   By default, the Pytorch and TensorFlow backends are built for CPU

   - Build default for CPU
      - cd Smart-Sim && source setup_env.sh
   - Build defaults for GPU
      - cd Smart-Sim && source setup_env.sh gpu
   - Build all backends for CPU
      - cd Smart-Sim && source setup_env.sh cpu 1 1 1 1
   - Build all backends for GPU
      - cd Smart-Sim && source setup_env.sh gpu 1 1 1 1


## Build SILC (optional)

  In order to build the scaling tests, use the SILC clients,
  or build the documentation, the SILC library must be installed

   - Get the SILC library
     - git clone #clone link to SILC#
     - git checkout develop
   - build the dependencies
     - source build_deps.sh

## Running the test suite


### Local

  If you are running the test suite locally on a laptop
  or workstation you can call pytest as follows

  ```bash
  export SMARTSIM_LOG_LEVEL=debug # optional, "develeper" is more verbose
  py.test -vv -o log_cli=true
  ```

  Stop pytest after the first failure to obtain the error and output files
  of the test by adding ``-x`` to the ``pytest`` command

### Slurm

  Users should usually obtain an allocation on a compute node
  so as to not take up space on the head node as the test suite
  runs a fair amount of tests.

  Obtain an allocation with 1 node and 1 process

  ```bash
  salloc -N 1 -n 1 -t 00:10:00
  ```

  run the test suite as follows
  ```bash
  conda activate smartsim-env # (optional) activate conda env
  srun -n 1 python -m pytest -vv --cov=../smartsim -o log_cli=true
  ```


