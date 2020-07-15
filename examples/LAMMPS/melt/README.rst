
*****************************************
Extracting Data from Compiled Simulations
*****************************************

This tutorial shows how to extract simulation data from a compiled
application using SmartSim client capabilities.  To demonstrate the
process in sufficient and concrete detail, the widely used molecular
dynamics (MD) code LAMMPS is used in this tutorial.  Using the
SmartSim C++ and Python clients, data will be streamed from a
LAMMPS simulation (written in C++) to a concurrently running
Python script.  In the following sections, the process of
embedding the client into the compiled application,
building a data processing script to retrieve simulation data,
and writing a SmartSim experiment script will be described.

Embedding the C++ client
========================

In the simplest cases, the SmartSim C++ client can be included
with only a few lines of code.  In the case of LAMMPS, an effort
was made to integrate SmartSim client functionality such that
it is accessible through standard LAMMPS input files, and as a result,
this example includes source and header file additions to LAMMPS that
go beyond minimum embedding steps.
The source and header file additions will be described in this
section along with changes made to the LAMMPS CMAKE build file.
While specific implementation details will be described for LAMMPS,
an effort has been made to provide direction that is applicable to
all applications.

Source and Header Files
-----------------------

The SmartSim Python, C++, C, and Fortran client source and header
files can be found in the ``smartsim/clients`` directory under
the top-level SmartSim repository directory.  For the C++
client, the ``SmartSimClient`` object is defined in ``client.h``
and ``client.cc``.  ``SmartSimClient`` is the object
through which a C++ application can send and receive data
in a SmartSim experiment.  To send and receive data,
simply include the ``client.h`` header file in your source
file, instantiate a ``SmartSimClient`` object,
and use ``SmartSimClient`` member functions to send and receive
simulation data.

For LAMMPS, an effort was made to conform to LAMMPS developer
conventions.  Specifically, LAMMPS allows for developers to add
packages to LAMMPS to define custom ``dump`` (i.e. output) styles
for simulation output.  In this example, we have added a package
to LAMMPS to enable the ``atom/smartsim`` dump style.
The ``atom/smartsim`` dump style sends atom ID, atom type,
and atom position to the SmartSim experiment database using the
aforementioned ``SmartSimClient`` object.  The default ``atom``
dump style in LAMMPS would normally send this information to output
files.

The source and header files added to LAMMPS are
``dump_atom_smartsim.cpp`` and ``dump_atom_smartsim.h``, respectively.
The files should be placed in a direcotry called
``USER-SMARTSIM`` inside of the ``lammps/src`` directory to
conform to LAMMPS developer standards.

The majority of code in ``dump_atom_smartsim.cpp``
is devoted to placing atom information from
LAMMPS data structures into arrays that can be sent to
the SmartSim database.  As shown in dump_atom_smartsim.cpp,
there are only three unique SmartSim client function
calls necessary to send the atom position data to the SmartSim experiment
database:

1) ```SmartSimClient`` contructor to create the SmartSim client
2) ``put_array_int64()`` to send the atom ID and atom type arays
3) ``put_array_double()`` to send the atom position arrays

For completeness, the code snippet below has been extracted from
the LAMMPS source files to show the exact
client API calls that are necessary to send the atom id,
atom type, and atom coordinates to the SmartSim database.

.. code-block:: cpp

    /* Declare an integer array array_dims that is used for
       bookkeeping to track the number of elements in a dynamically
       allocated arrays.  It is initialized to n_local which is
       a LAMMPS variable tracking the number of atoms.
    */
    int* array_dims = new int[1];
    array_dims[0] = n_local;

    /* Declare a set of temporary integer and double arrays
       that are used to extract data from LAMMPS output
       data structures.  This step is specifc to LAMMPS
       code structure, and may not be applicable to other
       applications.
    */
    int* data_int = new int[n_local];
    double* data_dbl = new double[n_local];

    /* Send LAMMPS atom position information to the SmartSim
       database via the SmartSimClient object.  Note
       that the function call to _pack_buf_into_array()
       fills our temporary integer and double arrays with
       LAMMPS data, and is not generalizable to other
       applications.
    */
    int buf_len = n_cols*n_local;
    //Atom ID
    this->_pack_buf_into_array<int>(data_int, buf_len, 0, n_cols);
    key = this->_make_key("atom_id", rank);
    client.put_array_int64(key.c_str(), data_int, array_dims, 1);
    //Atom Type
    this->_pack_buf_into_array<int>(data_int, buf_len, 1, n_cols);
    key = this->_make_key("atom_type", rank);
    client.put_array_int64(key.c_str(), data_int, array_dims, 1);
    //Atom x position
    this->_pack_buf_into_array<double>(data_dbl, buf_len, 2, n_cols);
    key = this->_make_key("atom_x", rank);
    client.put_array_double(key.c_str(), data_dbl, array_dims, 1);
    //Atom y position
    this->_pack_buf_into_array<double>(data_dbl, buf_len, 3, n_cols);
    key = this->_make_key("atom_y", rank);
    client.put_array_double(key.c_str(), data_dbl, array_dims, 1);
    //Atom z position
    this->_pack_buf_into_array<double>(data_dbl, buf_len, 4, n_cols);
    key = this->_make_key("atom_z", rank);
    client.put_array_double(key.c_str(), data_dbl, array_dims, 1);

The ``atom/smartsim`` dump style generates a key for the
data based on the simulation time step number, the
MPI rank of the process, the key prefix provided
by the user in the LAMMPS input file, and the
quantity being sent (e.g. atom_id).  This is shown in the
above code snippet with repeated calls to the ``_make_key()``
function that has been implemented in LAMMPS.  The ``_make_key()``
function is shown below as an example of key generation, but
the other applications  will require the user to
write their own key generation scheme. Note that in the current
implementation of the ``atom/smartsim`` dump style, each quantity is
sent to the SmartSim database as a 1D array in the same order
as the 1D array enumerating atom IDs and each MPI process
sends its own data to the database.  However, the ``SmartSimClient``
array functions support n-dimensional arrays.

.. code-block:: cpp

   std::string DumpAtomSmartSim::_make_key(std::string var_name, int rank)
   {
     /* This function creates a key for the data being sent
        to the database.  The variable filename below is a LAMMPS
	variable read from the LAMMPS input file that we use as an
	optional string prepended to the key.  The optional
	string is followed by the MPI rank, time step, and data
	variable name spearated by _ characters.  Note that each
	MPI (rank) sends its own data, and as a result, has a unique
	data key.
     */
     std::string prefix(filename);
     std::string key = prefix + "_rank_" + std::to_string(rank) +
     "_tstep_" + std::to_string(update->ntimestep) + "_" +
     var_name;
     return key;
  }


Compiling with the SmartSim Client
----------------------------------

To use the SmartSim client at application runtime,
the client will need to built and linked with your
application.  To build the SmartSim client with your
application, you will need to include the following items:

1)  ``smartsim/utils/protobuf/`` and
    ``smartsim/clients`` should be added to your
    include directories when compiling.  It is best
    practice to use the environment variable ``SMARTSIMHOME``
    that is set when sourcing ``setup_env.sh`` to
    point to the top level SmartSim directory when
    adding these directories.  In CMAKE, this could be
    implemented as shown in the code snippet below.

.. code-block:: cmake

  include_directories($ENV{SMARTSIMHOME}/smartsim/clients/)
  include_directories($ENV{SMARTSIMHOME}/smartsim/utils/protobuf/)

2)  ``hiredis``, ``redis-plus-plus``, and ``protobuf``
    include directories should be added to your make file include paths.
    These packages and include directories are installed
    during the SmartSim installation and can be referenced
    using environment variables set by ``setup_env.sh``.
    For those applications that use CMAKE, the code snippet
    below shows how a user can include the aforementioned
    directories.

.. code-block:: cmake

  string(CONCAT HIREDIS_INCLUDE_PATH $ENV{HIREDIS_INSTALL_PATH} "/include/")
  string(CONCAT PROTOBUF_INCLUDE_PATH $ENV{PROTOBUF_INSTALL_PATH} "/include/")
  string(CONCAT REDISPP_INCLUDE_PATH $ENV{REDISPP_INSTALL_PATH} "/include/")
  include_directories(${HIREDIS_INCLUDE_PATH})
  include_directories(${PROTOBUF_INCLUDE_PATH})
  include_directories(${REDISPP_INCLUDE_PATH})

3) The SmartSim client source file and associated protobuf message
   description file should be added
   to your source file build list.  For applications
   that use CMAKE, the code snippet below shows how users
   can use the environment variables set by ``setup_env.sh`` to
   add these files to a CMAKE variable ``CLIENT_SRC`` that can
   be used later when building your application.

.. code-block:: cmake

   set(CLIENT_SRC $ENV{SMARTSIMHOME}/smartsim/clients/client.cc
	$ENV{SMARTSIMHOME}/smartsim/utils/protobuf/smartsim_protobuf.pb.cc)

4) Add the ``hiredis``, ``redis-plus-plus``, and ``protobuf`` libraries
   to the list of libraries that will be linked into your application.
   For applications that use CMAKE, the code snippet
   below shows how a user can include the aforementioned
   libraries into their make file using the environment variables
   defined by ``setup_env.sh``.  In the code snippet below,
   the aforementioned libraries are all stored in a CMAKE variable
   ``CLIENT_LIBRARIES`` which can be easily referenced when linking
   the application.

.. code-block:: cmake

   # Use environment variable HIREDIS_INSTALL_PATH to set
   # location of hiredis lib
   string(CONCAT HIREDIS_LIB_PATH $ENV{HIREDIS_INSTALL_PATH} "/lib")
   find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH)

   # Use environment variable PROTOBUF_INSTALL_PATH to set
   # location of protobuf lib
   string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
   find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH)

   # Use environment variable REDISPP_INSTALL_PATH to set
   # location of redis-plus-plus lib
   string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
   find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} NO_DEFAULT_PATH)

   # Store the three libraries in a variable CLIENT_LIBRARIES for easy linking
   set(CLIENT_LIBRARIES ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

For clarity, the aforementioned code snippets have been folded into a
working CMAKE file shown below that would build the SmartSim client
into an application called ``my_application``.

.. code-block:: cmake

  project(Example)

  set(CMAKE_BUILD_TYPE Release)

  cmake_minimum_required(VERSION 3.10)

  SET(CMAKE_CXX_STANDARD 11)

  # Add the SmartSim Client include directories using the
  # SMARTSIMHOME environment variable
  include_directories($ENV{SMARTSIMHOME}/smartsim/clients/)
  include_directories($ENV{SMARTSIMHOME}/smartsim/utils/protobuf/)

  # Add the third-party package include paths to the
  # project using the environment variables provided by SMARTSIM
  string(CONCAT HIREDIS_INCLUDE_PATH $ENV{HIREDIS_INSTALL_PATH} "/include/")
  string(CONCAT PROTOBUF_INCLUDE_PATH $ENV{PROTOBUF_INSTALL_PATH} "/include/")
  string(CONCAT REDISPP_INCLUDE_PATH $ENV{REDISPP_INSTALL_PATH} "/include/")
  include_directories(${HIREDIS_INCLUDE_PATH})
  include_directories(${PROTOBUF_INCLUDE_PATH})
  include_directories(${REDISPP_INCLUDE_PATH})

  # Use environment variable HIREDIS_INSTALL_PATH to set
  # location of hiredis lib
  string(CONCAT HIREDIS_LIB_PATH $ENV{HIREDIS_INSTALL_PATH} "/lib")
  find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH)

  # Use environment variable PROTOBUF_INSTALL_PATH to set
  # location of protobuf lib
  string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
  find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH)

  # Use environment variable REDISPP_INSTALL_PATH to set
  # location of redis-plus-plus lib
  string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
  find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} NO_DEFAULT_PATH)

  # Store the three third-party libraries in a variable
  # CLIENT_LIBRARIES for easy linking
  set(CLIENT_LIBRARIES ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

  # Set the source files for the SmartSim Client to variable
  # CLIENT_SRC for later compilation
  set(CLIENT_SRC $ENV{SMARTSIMHOME}/smartsim/clients/client.cc
  	$ENV{SMARTSIMHOME}/smartsim/utils/protobuf/smartsim_protobuf.pb.cc)


  # Build my application with the additional CLIENT_SRC files
  add_executable(my_application
 	my_application.cpp
	${CLIENT_SRC}
  )

  # Link my application with the additional CLIENT_LIBRARIES
  # libaries
  target_link_libraries(my_application
  	${CLIENT_LIBRARIES}
  )

Because the ``atom/smartsim`` dump style is implemented
as a LAMMPS package in order to conform to LAMMPS
programming practices, adaptations of the above instructions
were made for the LAMMPS integration.  These adaptations
are not necessarily instructive for applications beyond LAMMPS,
so they are only briefly described herein.  In the list below,
the line nubmers corresponding to changes in the LAMMPS
CMakeLists.txt file are given so that these adaptations
can be easily referenced.  However, the same basic compiling
structured described above is followed.

1) ``USER-SMARTSIM`` was added as an optional build package
   `(line 136)`
2) When building the ``USER-SMARTSIM`` package with LAMMPS,
   logic is needed to add SmartSim and SmartSim
   dependency include directories to cmake `(line 174-198)`.
3) The SmartSim client source files are added
   to the build list for LAMMPS `(line 412 - 417)`.

It is worth reiterating that the CMAKE examples presented here
rely on environment variables set by
the SmartSim ``setup_env.sh`` script.  Therefore,
``setup_env.sh`` should be run before trying to compile
with the SmartSim client.

To build LAMMPS with the aforementioned
``USER-SMARTSIM`` package and MPI capabitlies,
the following cmake command can be used:

.. code-block:: bash

   cmake ../ -DBUILD_MPI=yes -DPKG_USER-SMARTSIM=ON

The LAMMPS binary location should be added
to the PATH environment variable so that the
SmartSim experiment can find it.  Additionally,
it is recommended that the "stable" branch
of LAMMPS be used for this tutorial.


Experiment Setup
================

In this tutorial, the SmartSim experiment consists of a
SmartSim model entity for the LAMMPS simulation and a
SmartSim node entity to intake and plot atom position
information from the SmartSim database.  The experiment
is configured to utilize the Slurm launcher
and a 3 node KeyDB redis cluster database.  The
SmartSim experiment script is shown below with
comments to explain the experiment progression.
It is worth noting that the inclusion of the
SmartSim client in LAMMPS does not alter the
typical experiment flow that has been described in
other tutorials.  In fact, no details of the
C++ client utiliation in LAMMPS are necessary
in the SmartSim experiment script.


.. code-block:: bash

  from smartsim import Experiment
  import os

  # Define resource variables that we will
  # use to get and manage system resources
  lammps_compute_nodes = 2
  db_compute_nodes = 3
  anlaysis_compute_nodes = 1

  total_compute_nodes = lammps_compute_nodes +
                        db_compute_nodes +
                        analysis_compute_nodes

  # Create a SmartSim Experiment using the default
  # Slurm launcher backend
  experiment = Experiment("lammps_melt_analysis")

  # Fetch a compute resource allocation using SmartSim
  # Experiment API
  alloc = experiment.get_allocation(total_compute_nodes, ppn=ppn)


  # Define the run settings for the LAMMPS model that will
  # be subsequently created.
  lammps_settings = {
      "nodes": lammps_compute_nodes,
      "ppn" : ppn,
      "executable": "lmp",
      "exe_args": "-i in.melt",
      "alloc": alloc}

  # Define the run settings for the Python analysis script
  # that will be subsequently created
  analysis_settings = {
      "nodes": analysis_compute_nodes,
      "executable": "python smartsim_node.py",
      "exe_args": f"--ranks={lammps_compute_nodes*ppn} --time=250",
      "alloc": alloc}

  # Create the LAMMPS SmartSim model entity with the previously
  # defined run settings
  m1 = experiment.create_model("lammps_melt", run_settings=lammps_settings)

  # Attach the simulation input file in.melt to the entity so that
  # the input file is copied into the experiment directory when it is created
  m1.attach_generator_files(to_copy=[f"{os.getcwd()}/in.melt"])

  # Create the analysis SmartSim node entity with the
  # previously defined run settings
  n1 = experiment.create_node("lammps_data_processor",run_settings=analysis_settings)

  # Attach the analysis script to the SmartSim node entity so that
  # the script is copied into the experiment directory when the
  # experiment is generated.
  n1.attach_generator_files(to_copy=[f"{os.getcwd()}/smartsim_node.py"])

  # Create the SmartSim orchestrator object and database using the default
  # database cluster setting of three database nodes
  orc = experiment.create_orchestrator_cluster(alloc, overwrite=True)

  # Generate the experiment directory structure and copy the files
  # attached to SmartSim entities into that folder structure.
  experiment.generate()

  # Start the experiment
  experiment.start()

  # Poll the status of the SmartSim model and node in a blocking
  # manner until both are completed
  experiment.poll()

  # When the model and node are complete, stop the
  # orchestrator with the stop() call which will
  # stop all running jobs when no entities are specified
  experiment.stop()

  # Release our system compute allocation
  experiment.release()


LAMMPS input file
-----------------

The LAMMPS input file ``in.melt`` shown below
was edited to include the ``atom/smartsim`` dump style
(line 23).  It is worth noting that this input
file command will send atom position data
to the SmartSim database every 50 time steps.
Moreover, the last parameter in the input file
line "atoms" will be used as a key prefix for all
keys sent to the database.  This input file
was attached to the SmartSim model entity
in the experiment script show in the previous section
so that it is copied into the experiment
directory created by SmartSim.

.. code-block:: bash
  :linenos:

  # 3d Lennard-Jones melt

  units		lj
  atom_style	atomic

  lattice	fcc 0.8442
  region	box block 0 10 0 10 0 10
  create_box	1 box
  create_atoms	1 box
  mass		1 1.0

  velocity	all create 3.0 87287

  pair_style	lj/cut 2.5
  pair_coeff	1 1 1.0 1.0 2.5

  neighbor	0.3 bin
  neigh_modify	every 20 delay 0 check no

  fix		1 all nve

  dump		id all atom 50 dump.melt
  dump		smart_sim all atom/smartsim 50 atoms

  thermo	50
  run		250


LAMMPS data analysis in Python
------------------------------

The analysis script that retrieves the atom position information
from the SmartSim database is shown below.  In this analysis
script, SmartSim Python client API functions are used to retreive
data from the database.  Specifically, ``get_array_nd_int64``
and ``get_array_nd_float64`` are used to retreive the data.
The function names in the Python client are nearly identical
to the C, Fortran, and C++ client function names, except for
the addition of the "array_nd" substring indicating that for
array data types a numpy n-dimensional array is used.

The important steps in the analysis script below
that are applicable to all Python scripts that send and receive
data from the SmartSim database are as follows:

1)  Import the SmartSim Python ``Client`` object for use in the
    script with ``from smartsim import Client``.
2)  Initialize a SmartSim ``Client`` object with
    ``client = Client()`` or
    ``client = Client(cluster=False)``.  The optional argument
    ``cluster`` is by default ``True``, and indicates whether
    or not a cluster of database nodes is being used.
3)  Use SmartSim ``Client`` member functions calls like
    ``client.get_array_nd_int64(key, wait=True)``
    and ``client.get_array_nd_float64(key, wait=True)``
    to retreive data from the database.  Note that the variable
    ``key`` above needs to be set to a valid key present in the
    database.  Also, note that the optional parameter ``wait``
    in the Python client retrieval functions allow for the
    execution of the script to be blocked until the key appears
    in the database.

In this particular example, the atom positions that are
fetched from the database are plotted on a three-dimensional
plot.  As expected, the molecules should be uniformly
distributed in the domain.

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from smartsim import Client

  if __name__ == "__main__":

      # The command line argument "ranks" is used to
      # know how many MPI ranks were used to run the
      # LAMMPS simulation because each MPI rank will send
      # a unique key to the database.  This command line
      # argument is provided programmatically as a
      # run setting in the SmartSim experiment script.
      # Similarly, the command line argument "time"
      # is used to set which time step data will be
      # pulled from the database.  This is also set
      # programmatically as a run setting in the SmartSim
      # experiment script
      import argparse
      argparser = argparse.ArgumentParser()
      argparser.add_argument("--ranks", type=int, default=1)
      argparser.add_argument("--time", type=int, default=0)
      args = argparser.parse_args()

      n_ranks = args.ranks
      t_step = args.time

      # Initialize the SmartSim client object and indicate
      # that a database cluster is being used with
      # cluster = True
      client = Client(cluster=True)

      # Create empty lists that we will fill with simulation data
      atom_id = []
      atom_type = []
      atom_x = []
      atom_y = []
      atom_z = []

      # We will loop over MPI ranks and fetch the data
      # associated with each MPI rank at a given time step.
      # Each variable is saved in a separate list.
      for i in range(n_ranks):
          key = f"atoms_rank_{i}_tstep_{t_step}_atom_id"
          print(f"loking for key {key}")
          atom_id.extend(client.get_array_nd_int64(key, wait=True))
          key = f"atoms_rank_{i}_tstep_{t_step}_atom_type"
          atom_type.extend(client.get_array_nd_int64(key, wait=True))
          key = f"atoms_rank_{i}_tstep_{t_step}_atom_x"
          atom_x.extend(client.get_array_nd_float64(key, wait=True))
          key = f"atoms_rank_{i}_tstep_{t_step}_atom_y"
          atom_y.extend(client.get_array_nd_float64(key, wait=True))
          key = f"atoms_rank_{i}_tstep_{t_step}_atom_z"
          atom_z.extend(client.get_array_nd_float64(key, wait=True))

      # We print the atom position data to check the accuracy of our results.
      # The printed data will be piped by SmartSim to an output file
      # in the experiment directory.
      n_atoms = len(atom_id)
      for i in range(n_atoms):
          print(f"{atom_id[i]} {atom_type[i]} {atom_x[i]} {atom_y[i]} {atom_z[i]}")

      # We plot the atom positions to check that the atom position distribution
      # is uniform, as expected.
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      ax.set_title('Atom position')
      ax.scatter(atom_x, atom_y, atom_z)
      plt.savefig('atom_position.pdf')


Running the Experiment
----------------------

They SmartSim experiment can be run with the
command:

.. code-block:: bash

   python run-melt.py
