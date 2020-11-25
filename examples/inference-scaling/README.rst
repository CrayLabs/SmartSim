
*************************
Inference Scaling Example
*************************

The example within this directory performs a batch of parallel inference
tests with a Pytorch model (either MNIST or Resnet) with a single
image corresponding to the dataset on which the model was trained

The test currently runs the clients as a C++ MPI program using the
C++ SILC client to perform the tensor commands to the Redis Database.

The script can launch multiple sequential inference sessions on the
same allocations.

Note that the following examples are written to be run on a Slurm
system equipped with GPUs.

Training the Models
===================

There are two models in the inference scaling example section:

  1. Resnet50 trained on Imagenet
  2. CNN trained on the MNIST dataset

To train the MNIST model, use the script ``mnist.py``

.. code-block:: bash

    python mnist.py
    cp mnist_cnn.pt mnist_data

To obtain the trained Resnet model, call the following

.. code-block:: bash

    python model_saver.py
    cp resnet50.pt imagenet

Both of these scripts are currently written to run on a GPU
with the CUDA libraries installed. Both examples can be
changed to run on CPU by adapting these scripts.


Running the Scaling Tests
=========================

The following will walk through downloading silc, building the example,
and setting the parameters for the test. Prior to the start, make sure
that the ``setup_env.sh`` script has been sourced and a virtual
environment with the SmartSim requirements is active.

Downloading and Building SILC
-----------------------------

First, SILC needs to be downloaded and setup in the SMARTSIMHOME
directory. this is the top level directory for SmartSim.

.. code-block:: bash

    git clone <clone link to SILC>
    git checkout develop
    source build_deps.sh

The last line will build and install all the third-party packages
necessary for SILC. Make sure to have your compiler toolchain
and programming environment setup before this is called.

Building the Scaling Tests
--------------------------

Next, we build the scaling tests themselves with SILC C++ client
included.

This will build both the Resnet and MNIST examples

.. code-block:: bash

    cd examples/inference-scaling/
    mkdir build
    cd build
    CC=cc CXX=CC cmake .. # for Cray machines
    make

Set the Experiment Parameters
-----------------------------

There are two places to set the experiment parameters

    1) Inside the C++ program we just built
    2) Inside the SmartSim script that runs the tests.

The ``inference-scaling.cpp`` program includes two places
where arguments can be changed for the scaling tests.

 - Batch Size (controls the size of inference batch)
 - Number of iterations (number of inferences performed per run)

the batch size is currently set to ``10`` and the number
of iterations (set on line 67) is set to ``50``.

The SmartSim Script also includes parameters for the
inference scaling tests at the top of the file.

.. code-block:: python

    # Constants
    DB_NODES = 3              # number of database nodes
    DPN = 1                   # number of databases per node
    CLIENT_ALLOC = 40         # number of nodes in client alloc
    CLIENT_NODES = [20, 40]   # list of node sizes to run clients within client alloc
    CPN = [80]                # clients per node
    NAME = "infer-scaling"    # name of experiment directory

In the current setup, 2 runs of 1600 (80 x 20) and 3200 (80 x 40) clients
respectively will be executed. Since there are 50 iterations of inference
per client, these runs will execute 80000 and 160000 inferences respectively.

All runs will execute on the same allocations which can be customized in the
script to obtain whatever allocation suits your machine.

Post-Processing
---------------

The inference program in C++ will output a number of CSV files for each MPI rank
that contain timings for the inference workload. these times are collected by a
post-processing script that collects the results into a single CSV that includes
the experiemnt summary.

The inference statistics will be under <NAME>.csv after a successful run.

Execution
=========

To submit the inference scaling tests to slurm (or pbs if you change the launcher),
simply run the SmartSim script corresponding to the model you would
like the scaling results of.

For example, for MNIST

.. code-block:: bash

    python run-inference-session.py

or for Resnet

.. code-block:: bash

    python run-inference-session-resnet.py
