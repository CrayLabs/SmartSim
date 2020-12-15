
*******
Testing
*******

This document describes how to run the tests for SmartSim
on various machines.

.. note::

  For the test to run, you must have the ``requirements-dev.txt``
  dependencies installed in your python environment.


Running the Test Suite
======================

Local
-----

If you are running the test suite locally on a laptop
or workstation you can call pytest as follows in the
top level of the SmartSim directory.

.. code-block:: bash

  export SMARTSIM_LOG_LEVEL=debug # optional, "developer" is more verbose
  make test
  make test-verbose # add verbosity
  make test-cov # test with coverage


Slurm
-----

Users should usually obtain an allocation on a compute node
so as to not take up space on the head node as the test suite
runs a fair amount of tests.

For testing on slurm, users will not be able to use the built-in
makefile. Pytest will have to be invoked manually through the
command line as follows:

Obtain an allocation with 1 node and 1 process

.. code-block:: bash

  # obtain an allocation
  salloc -N 1 -n 1 -t 00:10:00

  # run the test suite
  conda activate smartsim-env # (optional) activate conda environment
  srun -n 1 python -m pytest -vv --cov=../smartsim -o log_cli=true
