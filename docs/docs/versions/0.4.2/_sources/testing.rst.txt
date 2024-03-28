*******
Testing
*******

This is an overview documentation for testing SmartSim and SmartRedis

SmartSim
========

SmartSim utilizes ``pytest`` for running its test suite. In the
top level of SmartSim, users can run multiple testing commands
with the developer Makefile.

To run, execute ``make`` plus one of the below commands from the top
level of the SmartSim directory::

  Test
  -------
  test                - Build and run all tests
  test-verbose        - Build and run all tests [verbosely]
  test-debug          - Build and run all tests with debug output
  test-cov            - Run tests with coverage
  test-full           - Run all WLM tests with Python coverage (full test suite)
                        WARNING: do not run test-full on shared systems.

For the test to run, you must have the ``requirements-dev.txt``
dependencies installed in your python environment.

There are two ways to do this
1. Install smartsim with ``dev`` extension ``pip install -e .[dev]``
2. ``pip install -r requirements-dev.txt`` after you install smartsim.


Test Suites
-----------

There are three test suite levels within SmartSim
  - ``local``
  - ``on_wlm``
  - ``full``

``local``
~~~~~~~~~

``local`` runs by default and doesn't launch any jobs out onto
a system through a workload manager like Slurm. All jobs are contained
on the local machine.

This is the test suite that runs in GitHub actions each time a commit
is used.

To run the local tests

.. code:: bash

  bash
  make test
  # or
  make test-cov # for coverage
  # or
  make test-debug # for CLi logging output

``on_wlm``
~~~~~~~~~~

This is the same test suite as the local test suite with the addition
of the tests located within the ``on_wlm`` directory.

To run the ``on_wlm`` test suite, users will have to be on a system
with one of the supported workload managers. Additionally, users will
need to obtain an allocation of **at least 3 nodes**.

Examples of how to obtain allocations on systems with the launchers:

.. code:: bash

  # for slurm (with srun)
  salloc -N 3 -A account --exclusive -t 00:10:00

  # for PBSPro (with aprun)
  qsub -l select=3 -l place=scatter -l walltime=00:10:00 -q queue

  # for Cobalt (with aprun)
  qsub -n 3 -t 00:10:00 -A account -q queue -I

  # for LSF (with jsrun)
  bsub -Is -W 00:30 -nnodes 3 -P project $SHELL

Values for queue, account, or project should be substituted appropriately.

Once in an iterative allocation, users will need to set the test
launcher environment variable: ``SMARTSIM_TEST_LAUNCHER`` to one
of the following values

 - slurm
 - cobalt
 - pbs
 - lsf
 - local

In addition to the ``SMARTSIM_TEST_LAUNCHER`` variable, there
are a few other runtime test configuration options for SmartSim

 - ``SMARTSIM_TEST_LAUNCHER``: Workload manager of the system (local by default)
 - ``SMARTSIM_TEST_ACCOUNT``: Project account for allocations (used for customer systems mostly)
 - ``SMARTSIM_TEST_DEVICE``: ``cpu`` or ``gpu``
 - ``SMARTSIM_TEST_INTERFACE``: network interface to use.

For the ``SMARTSIM_TEST_INTERFACE``, the default is ``ipogif0`` which
is the high speed network on Horizon, and other XC systems with the Aries
interconnect.

Other possible values are:
 - ``ipogif0``
 - ``ib0`` (and other ib variants)
 - ``eth0``

For the local test suite, the network interface does not need
to be set.


A full example on an internal SLURM system

.. code:: bash

  salloc -N 3 -A account --exclusive -t 03:00:00
  export SMARTSIM_TEST_LAUNCHER=slurm
  export SMARTSIM_TEST_INTERFACE=ipogif0
  export SMARTSIM_TEST_DEVICE=gpu
  make test-debug

``full_wlm``
~~~~~~~~~~~~

The full test suite runs the ``on_wlm`` tests in addition to tests
that will allocate and run on their own allocations. This is the only
way that the batch interface is tested.

Unless you know what you're doing, **do not run this on customer systems**

Writing Tests for SmartSim

When you introduce new code, it's imperative that tests accompany your PR.
Below are some guidelines for writing new tests.

 - All test files start with ``test_``
 - All test functions start with ``test_``
 - Function name should signal whats being tested
 - All static test files should go in ``SmartSim/tests/test_configs``
 - All test output should be located in ``SmartSim/tests/test_output`` (see below on ``conftest.py``)

Write most tests within the base ``SmartSim/tests`` directory unless they
are meant to specifically test a launcher integration that necessitates its
placement into the ``on_wlm`` or the ``full_wlm`` directory.

Any tests that run AI/ML tests for the backend should be placed in the
``SmartSim/tests/backend`` directory.

Most tests utilize the convenience functions with ``conftest.py`` which
can be added as arguments to test functions to supply runtime values
for test directories and what not. See tests for examples.

SmartRedis
==========

Before building the tests, it is assumed that the base dependencies
for SmartRedis described in the installation instructions have already
been executed.

Test Suites
-----------

There are two test suites for SmartRedis

  - ``Integration``
  - ``Unit``

Both test suites are run together.

Building and Running the Test Suites
------------------------------------

To build the tests, you first need to install the dependencies for
testing. To download SmartRedis related testing dependencies, run
the following:

.. code:: bash

  make test-deps
  # or to run tests on GPU hardware:
  make test-deps-gpu

The test suite is currently written to be run on CPU hardware to
test model and script executions.  Testing on GPU hardware
currently requires modifications to the test suite.

The tests require
- GCC > 5
- CMake > 3

Since these are usually system libraries we do not install them
for the user

Setting up Test Environment and Redis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


After installing dependencies and setting up your testing environment with
``setup_test_env.sh``, all tests can be built with the following command:

.. code:: bash

  ./setup_test_env.sh
  make build-tests

Before running the tests, users will have to spin up a Redis
cluster instance and set the ``SSDB`` environment variable.

To spin up a local Redis cluster, use the script
in ``utils/create_cluster`` as follows:

.. code:: bash

  cd path/to/smartredis          # navigate to the top level dir of smartredis
  conda activate YOUR_CONDA_ENV  # activate python env with SmartRedis requirements
  source setup_test_env.sh       # Setup smartredis environment
  cd utils/create_cluster
  python local_cluster.py        # spin up Redis cluster locally
  export SSDB="127.0.0.1:6379,127.0.0.1:6380,127.0.0.1:6381"  # Set database location

A similar script ``utils/create_cluster/slurm_cluster.py``
assists with launching a Redis cluster for testing on
Slurm managed machines.  This script has only been tested
on a Cray XC, and it may not be portable to all machines.

Running the Tests
~~~~~~~~~~~~~~~~~

If you are running the tests in a new terminal from the
one used to build the tests and run the Redis cluster,
remember to load your python environment with SmartRedis
dependencies, source the ``setup_test_env.sh`` file,
and set the ``SSDB`` environment variable.

To build and run all tests, run the following command in the top
level of the smartredis repository.

.. code:: bash

  make test

You can also run tests for individual clients as follows::

  test                           - Build and run all tests (C, C++, Fortran, Python)
  test-verbose                   - Build and run all tests [verbosely]
  test-c                         - Build and run all C tests
  test-cpp                       - Build and run all C++ tests
  unit-test-cpp                  - Build and run unit tests for C++
  test-py                        - run python tests
  test-fortran                   - run fortran tests
  testpy-cov                     - run python tests with coverage
  testcpp-cov                    - run cpp unit tests with coverage

Tearing down the Test Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To stop Redis, use the following commands

.. code:: bash

  cd utils/create_cluster
  python local_cluster.py --stop # stop the Redis cluster

The same works for the Slurm variant, but you can also just
release the allocation which is easier.

Writing Tests for SmartRedis
----------------------------

Below are some guidelines for writing new tests. These are fairly similar
to SmartSim.

 - Place tests for each client in their language directory (i.e. tests for C client in ``SmartRedis/tests/c``)
 - All test files start with ``test_``
 - All test functions start with ``test_``
 - Function name should signal whats being tested

Writing Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

The integrations tests are run with the ``pytest`` framework and some
helper python files that spin up the client drivers. Follow the naming
convention above and the tests will be included.

Writing Unit Tests
~~~~~~~~~~~~~~~~~~

All unit tests for the C++ client are located at ``tests/cpp/unit-tests/`` and use the Catch2
test framework. The unit tests mostly follow a Behavior Driven Development (BDD) style by
using Catch2's ``SCENARIO``, ``GIVEN``, ``WHEN``, and ``THEN`` syntax.

Files that contain Catch2 unit tests should be prefixed with *test_* in order to keep a
consistent naming convention.

When adding new unit tests, create a new ``SCENARIO`` in the appropriate file. If no such
file exists, then it is preferred that a new file (prefixed with *test_*) is created.


  - New unit tests should be placed in ``tests/cpp/unit-tests/``
  - Testing files should be prefixed with *test_*
  - It is preferred that new unit tests are in a new ``SCENARIO``

Continuous Integration (CI)
===========================

GitHub Actions is our public facing CI that runs in the GitHub cloud.

The actions are defined using yaml files are are located in the
``.github/workflows/`` directory of SmartSim and SmartRedis.

Each pull request, push and merge the test suite for SmartRedis
and SmartSim are run. For SmartSim, this is the ``local`` test suite
with the local launcher.

