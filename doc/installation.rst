****************
Install SmartSim
****************

From Source
===========

If your system does not include SmartSim
as a module, the instructions in this section
can be used to build and install SmartSim from source.

Prerequisites
-------------
The prerequisites to begin building SmartSim are:

- Python 3.7.x (or later)
- C compiler
- C++ compiler
- CMake 3.10.x (or later)

Python libraries
----------------

Most SmartSim library requirements can be installed via Python's
package manager ``pip`` with the following command executed in the
top level directory of SmartSim.

.. code-block:: bash

  pip install -r requirements.txt

If users wish to install these requirements manually, the packages
installed by the previous *pip* command are shown below.  Care
should be taken to match package version numbers to avoid build
and runtime errors.

- coloredlogs==10.0
- pytest==5.0.1
- toolz==0.10.0
- decorator==4.4.2
- redis==3.0.1
- redis-py-cluster==2.0.0
- breathe==4.19.2
- sphinx==3.1.1
- numpy>=1.18.2
- sphinx_rtd_theme>=0.5.0
- tqdm>=4.50.2
- psutil>=5.7.2
- pandas>=1.1.3


Third Party Libraries
---------------------

KeyDB_, `Redis`_, and RedisAI_ are also required
in order to use all features of SmartSim.  These packages
can be downloaded, compiled, and installed by executing the
following command in the top level of the SmartSim project:

.. _KeyDB: https://github.com/JohnSully/KeyDB
.. _Redis: https://github.com/redis/redis
.. _RedisAI: https://github.com/RedisAI/RedisAI

.. code-block:: bash

  ./setup_env.sh

The ``setup_env.sh`` script will install the three packages into
the ``third-party`` directory in the top level directory of
SmartSim.  The ``setup_env.sh`` script will print all build and
and installation output to the terminal, and any failures
will be evident there.  These packages do not require
many dependencies, but it is worth checking that
your system meets the prerequisites
listed on each project GitHub page.  It is not recommended
that users install these packages without using ``setup_env.sh``
because specific versions and build settings
have been selected for SmartSim.

In addition to installing packages, ``setup_env.sh`` sets
system environment variables that are used by SmartSim
to run experiments and can be used by the user to
locate files needed to  build SmartSim clients into their
applications.  The use of environment variables for compiling
SmartSim clients into applications is discussed in the client
documentation. The user should source ``setup_env.sh`` whenever
beginning a new session to ensure that environment
variables are properly set.


Suggested Environments for Cray Systems
=======================================

SmartSim has been built and tested on a number of
Cray XC and CS systems.  To help users build from source
on the systems that have been tested, in this section
the terminal commands to load a working environment
will be given.

Osprey (Cray CS)
----------------

.. code-block:: console

		module load PrgEnv-cray/1.0.6
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load cmake/3.10.2
		module load gcc/8.1.0
		module load cudatoolkit/10.1

Loon (Cray CS)
--------------

.. code-block:: console

		module load PrgEnv-cray/1.0.6
		module unload cray-libsci/17.12.1
		module load cmake/3.10.2
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Raptor (Cray CS)
----------------

.. code-block:: console

		module load PrgEnv-cray/1.0.6
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load cmake/3.10.3
		module load gcc/8.1.0

Tiger (Cray XC)
---------------

.. code-block:: console

		module load PrgEnv-cray/6.0.7
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Jupiter (Cray XC)
-----------------

.. code-block:: console

		module load PrgEnv-cray/6.0.7
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Heron
-----

.. code-block:: console

		module load PrgEnv-cray/6.0.7
		export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
		module load gcc/8.1.0

Cicero (Cray XC)
----------------

*Default system configurations and modules are sufficient on Cicero.*
