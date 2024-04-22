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

.. note::

  You must have the extra dev dependencies installed in your python environment to
  execute tests. Install ``dev`` dependencies with ``pip install -e .[dev]``


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
need to obtain an allocation of **at least 4 nodes**.

Examples of how to obtain allocations on systems with the launchers:

.. code:: bash

  # for slurm (with srun)
  salloc -N 4 -A account --exclusive -t 00:10:00

  # for PBSPro (with aprun)
  qsub -l select=4 -l place=scatter -l walltime=00:10:00 -q queue

  # for LSF (with jsrun)
  bsub -Is -W 00:30 -nnodes 4 -P project $SHELL

Values for queue, account, or project should be substituted appropriately.

Once in an iterative allocation, users will need to set the test
launcher environment variable: ``SMARTSIM_TEST_LAUNCHER`` to one
of the following values

 - slurm
 - pbs
 - lsf
 - local

In addition to the ``SMARTSIM_TEST_LAUNCHER`` variable, there
are a few other runtime test configuration options for SmartSim

 - ``SMARTSIM_TEST_LAUNCHER``: Workload manager of the system (local by default)
 - ``SMARTSIM_TEST_ACCOUNT``: Project account for allocations (used for customer systems mostly)
 - ``SMARTSIM_TEST_DEVICE``: ``cpu`` or ``gpu``
 - ``SMARTSIM_TEST_NUM_GPUS``: the number of GPUs to use for model and script testing (defaults to 1)
 - ``SMARTSIM_TEST_PORT``: the port to use for database communication (defaults to 6780)
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

  salloc -N 4 -A account --exclusive -t 03:00:00
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

To build and run all tests, run the following command in the top
level of the smartredis repository.

.. code:: bash

  make test

.. note::

  The tests require:
   - GCC >= 5
   - CMake >= 3.13

  Since these are usually system libraries, we do not install them
  for the user.


.. code-block:: bash

  make test-c         # run C tests
  make test-fortran   # run Fortran tests. Implicitly, SR_FORTRAN=ON
  make test-cpp       # run all C++ tests
  make unit-test-cpp  # run unit tests for C++
  make test-py        # run Python tests. Implicitly, SR_PYTHON=ON
  make testpy-cov     # run python tests with coverage. Implicitly, SR_PYTHON=ON SR_BUILD=COVERAGE
  make testcpp-cpv    # run cpp unit tests with coverage. Implicitly, SR_BUILD=COVERAGE


Customizing the Tests
---------------------

Several Make variables can adjust the manner in which tests are run:
   - SR_BUILD: change the way that the SmartRedis library is built. (supported: Release, Debug, Coverage; default for testing is Debug)
   - SR_FORTRAN: enable Fortran language build and testing (default is OFF)
   - SR_PYTHON: enable Python language build and testing (default is OFF)
   - SR_TEST_PORT: change the base port for Redis communication (default is 6379)
   - SR_TEST_NODES: change the number of Redis shards used for testing (default is 3)
   - SR_TEST_REDIS_MODE: change the type(s) of Redis servers used for testing. Supported is Clustered, Standalone, UDS; default is Clustered)
   - SR_TEST_REDISAI_VER: change the version of RedisAI used for testing. (Default is v1.2.3; the parameter corresponds the the RedisAI gitHub branch name for the release)
   - SR_TEST_DEVICE: change the type of device to test against. (Supported is cpu, gpu; default is cpu)
   - SR_TEST_PYTEST_FLAGS: tweak flags sent to pytest when executing tests (default is -vv -s)

These variables are all orthogonal. For example, to run tests for all languages against
a standalone Redis server, execute the following command:

.. code-block:: bash

  make test SR_FORTRAN=ON SR_PYTHON=ON SR_TEST_REDIS_MODE=Standalone

Similarly, it is possible to run the tests against each type of Redis server in sequence
(all tests against a standalone Redis server, then all tests against a Clustered server,
then all tests against a standalone server with a Unix domain socket connection) via the
following command:

.. code-block:: bash

  make test SR_FORTRAN=ON SR_PYTHON=ON SR_TEST_REDIS_MODE=All

.. note::

  Unix domain socket connections are not supported on MacOS. If the SmartRedis test
  system detects that it is running on MacOS, it will automatically skip UDS testing.

Writing Tests for SmartRedis
----------------------------

Below are some guidelines for writing new tests. These are fairly similar
to SmartSim.

 - Place tests for each client in their language directory (i.e. tests for C client in ``SmartRedis/tests/c``)
 - All test files start with ``test_``
 - All test functions start with ``test_``
 - Function name should signal what's being tested

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
