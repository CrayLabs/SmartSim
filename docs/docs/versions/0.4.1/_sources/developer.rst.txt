
*********
Developer
*********

This section details common practices and tips for contributors
to SmartSim and SmartRedis.

================
Testing SmartSim
================

.. note::

    This section describes how to run the SmartSim (infrastructure library)
    test suite. For testing SmartRedis, see below

SmartSim utilizes ``Pytest`` for running its test suite. In the
top level of SmartSim, users can run multiple testing commands
with the developer Makefile

.. code-block:: text

    Test
    -------
    test                       - Build and run all tests
    test-verbose               - Build and run all tests [verbosely]
    test-cov                   - run python tests with coverage

.. note::

  For the test to run, you must have the ``requirements-dev.txt``
  dependencies installed in your python environment.


Local
=====

There are two levels of testing in SmartSim. The first
runs by default and does not launch any jobs out onto
a system through a workload manager like Cobalt.

If any of the above commands are used, the test suite will
run the "light" test suite by default.


PBSPro, Slurm, Cobalt, LSF
==========================

To run the full test suite, users will have to be on a system
with one of the above workload managers. Additionally, users will
need to obtain an allocation of at least 3 nodes.

.. code-block:: bash

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

If tests have to run on an account or project, 
the environment variable ``SMARTSIM_TEST_ACCOUNT`` can be set.

-------------------------------------------------------

==================
Testing SmartRedis
==================

.. include:: ../smartredis/doc/testing.rst
   :start-line: 3


============
Git Workflow
============

Setup
=====

  - Fork the SmartSim (SmartRedis) repository
  - The origin remote should be set to your fork for pull and push
  - Set upstream as the main repository and set upstream push remote to ``no_push``
  - Follow installation instructions


Pull Requests
=============

Please check the following before submitting a pull request to the SmartSim repository

  1) Your feature is on a new branch off master.
  2) You are merging the feature branch from your fork into the main repository.
  3) All unnecessary whitespace has been purged from your code.
  4) For Python code changes, Black and isort have been applied to format code and sort imports.
  5) Pylint errors have been minimized as much as possible.
  6) All your code has been appropriately documented.
  7) The PR description is clear and concise.
  8) You have requested a review.

Merging
=======

When merging, there are a few guidelines to follow

   - Wrap all merge messages to 70 characters per line.


-------------------------------------------------------


=================
Python Guidelines
=================

For the most part, the conventions are to follow PEP8 that is supplied by pylint. However, there
are a few things to specifically mention.

  - Underscores should precede methods not meant to be used outside a class
  - All methods should have docstrings (with some exceptions)
  - Variable names should accurately capture what values it is storing without being overly verbose
  - No bad words
  - Use Black and isort frequently
  - Utilize ``conftest.py`` for creating pytest fixtures


SmartSim Python Style Guide Do's and Don'ts:

  - DON'T use global variables or the global keyword unless necessary
  - DON'T over comment code when it reads like English
  - DO use underscores on methods that should not be used outside a class
  - DO use comprehensions
  - DON'T write functions for more than one purpose
  - DON'T allow functions to return more than one type
  - DON'T use protected member variables outside a class
  - DON'T use single letter variable names
  - DO use entire words in names (i.e. get_function not get_func)
  - DON'T use wildcard imports (i.e. ``from package import *``) unless ``__all__`` is defined
  - DO use snake_case when naming functions, methods, and variables
  - DO use PascalCase when naming Classes
  - DO use the logging module

---------------------------------------------------------

==================
Editor Suggestions
==================

The editor that we suggest developers use is VSCode. Below are some extensions that
could make the process of developing on SmartSim a bit easier.

    - GitLens, for viewing changes in the git history
    - Remote SSH, for connecting to clusters and supercomputers
    - PyLance, Language Server
    - Python indent, for correcting python indents
    - reStructuredText, for writing documentation
    - Strict Whitespace, for ensuring no whitespace left in code
    - Python Docstring Generator, for writing docstring quickly
    - C/C++, for client development
    - Settings Sync, for syncing settings across remote servers