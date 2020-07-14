
*****************************************
Extracting Data from Compiled Simulations
*****************************************


This example shows the integration of the SmartSim C++ client with the
LAMMPS molecular dynamics (MD) code to send data from a LAMMPS simulation
to a python analysis script.  In this doc, the process of incorporating
the SmartSim client into LAMMPS will be documented and the SmartSim
experiment that runs the LAMMPS simulation and analysis script will be
described.

Embedding the C++ client
========================

In the simplest cases, the SmartSim C++ client can be included
with only a few lines of code.  In the case of LAMMPS, an effort
was made to integrate SmartSim client functionality such that
it is accessible through standard LAMMPS input files, and as a result,
this example includes source and header file additions to LAMMPS.
The source and header file additions will be described in this
section along with changes made to the LAMMPS cmake build file.

Source and Header Files
-----------------------

LAMMPS allows for developers to add packages to LAMMPS to define
custom ``dump`` styles for simulation output.  In this example,
we have added a package to LAMMPS to enable the
``atom/smartsim`` dump style.  The ``atom/smartsim`` dump style
sends atom ID, atom type, and atom position to the
SmartSim experiment database.  The default ``atom`` dump style
in LAMMPS would normally send this information to output
files.

The source and header files added to LAMMPS are
dump_atom_smartsim.cpp and dump_atom_smartsim.h, respectively.
The files should be placed in a direcotry called
``USER-SMARTSIM`` inside of the ``lammps/src`` directory to
conform to LAMMPS developer standards.

The majority of the code in dump_atom_smartsim.cpp
is devoted to placing atom information from
LAMMPS data structures into arrays that can be sent to
the SmartSim database.  As shown in dump_atom_smartsim.cpp,
there are only three unique SmartSim client function
calls necessary to send the data to the SmartSim experiment
database:

1) SmartSim client contructor to create the SmartSim client
2) ``put_array_int64()`` to send the atom ID and atom type arays
3) ``put_array_double()`` to send the atom position arrays

The ``atom/smartsim`` dump style generates a key for the
data based on the simultion time step number, the
MPI rank of the process, the key prefix provided
by the user in the LAMMPS input file, and the
quantity being sent (e.g. atom_id).  Note that in the
current implementation of the ``atom/smartsim`` dump
style, each quantity is sent to the SmartSim
database as a 1D array in the same order
as the 1D array enumerating atom IDs.

LAMMPS cmake updates
--------------------

Because the ``atom/smartsim`` dump style is implemented
as a LAMMPS package, updates need to be made to the
LAMMPS cmake build file.  For clarity, line numbers
corresponding to changes in the included
CMakeLists.txt file will be given as updates are
enumerated.  The following updates were made to
the cmake build file:

1) ``USER-SMARTSIM`` was added as an optional build package
   `(line 136)`
2) When building the ``USER-SMARTSIM`` package with LAMMPS,
   logic is needed to add SmartSim and SmartSim
   dependency include directories to cmake `(line 174-198)`.
3) The SmartSim client source files are added
   to the build list for LAMMPS `(line 412 - 417)`.

It is important to note that the updates to the LAMMPS
cmake file rely on environment variables set by
the SmartSim setup_env.sh script.  Therefore,
setup_env.sh should be run before trying to compile
LAMMPS with the SmartSim client.

To build LAMMPS with the aforementioned
``USER-SMARTSIM`` package and MPI capabitlies,
the following cmake command can be used:

.. code-block:: bash

   cmake ../ -DBUILD_MPI=yes -DPKG_USER-SMARTSIM=ON

The LAMMPS binary location should be added
to the PATH environment variable so that the
SmartSim experiment can find it.  Additionally,
it is recommended that the "stable" branch
of LAMMPS be used for this example.


Experiment Setup
================

The SmartSim experiment consists of a SmartSim model
entity for the LAMMPS simulation and a SmartSim node
entity to intake and plot atom position informtion
from the SmartSim database.  The experiment
is configured to utilize the Slurm launcher
and a 3 node KeyDB redis cluster database.

LAMMPS input files
--------------------------

The LAMMPS input file ``in.melt`` was edited
to include the ``atom/smartsim`` dump stytle with
the following command line:

.. code-block:: bash

   dump      ss all atom/smartsim 50 atoms

The command above creates a simulation data dump
at every 50 time steps.

Configuring the Experiment
--------------------------

show python code and talk through it.


Setting up the Analysis
-----------------------

describe setting up the analysis
show python code.


Running the Experiment
----------------------

They SmartSim experiment can be run with the
command:

.. code-block:: bash

   python run-melt.py
