
*****************************************
Extracting Data from Compiled Simulations
*****************************************

.. note::
   This example uses API functionality in the 0.3.0-beta
   release of SmartSim and 0.1.0-alpha relase of SILC.

This tutorial shows how to extract simulation data from a compiled
application using SILC client capabilities.  To demonstrate the
process in sufficient and concrete detail, the widely used molecular
dynamics (MD) code LAMMPS is used in this tutorial.  Using the
SILC C++ and Python clients, data will be streamed from a
LAMMPS simulation (written in C++) to a data processing
Python script.  In the following sections, the process of
embedding the client into the compiled application,
building a data processing script to retrieve simulation data,
and writing a SmartSim experiment script will be described.

Embedding the C++ client
========================

In the simplest cases, the SILC C++ client can be included
with only a few lines of code.  In the case of LAMMPS, an effort
was made to integrate SILC client functionality such that
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

The SILC library  consists of two user-facing objects:
``Client`` and ``DataSet``.  The ``Client`` object is
required to execute data, script, and ML/AI model
tasks.  The ``DataSet`` object provides the user with
a nested data structure that encapsulates tensors and
metadata, and the ``DataSet`` tensors can be used in the
aforementioned ``Client`` tasks.

The Python, C++, C, and Fortran client source and header
files can be found in the ``SILC`` repository.  For the C++
client, the ``Client`` object is defined in ``client.h``
and ``client.cpp``.  Similarly, the C++ ``DataSet`` object
is defined in ``dataset.h`` and ``dataset.cpp``.  Since
LAMMPS is written in C++, the C++ interface in the
aforementioned files will be the focus of this tutorial.
However,

For LAMMPS, an effort was made to conform to LAMMPS developer
conventions.  Specifically, LAMMPS allows for developers to add
packages to LAMMPS to define custom ``dump`` (i.e. output) styles
for simulation output.  In this example, we have added a package
to LAMMPS to enable the ``atom/smartsim`` dump style.
The ``atom/smartsim`` dump style sends atom ID, atom type, atom
position, and other simulation metadata to the
SmartSim experiment database using the
aforementioned ``SmartSimClient`` object.  The default ``atom``
dump style in LAMMPS would normally send this information to
output files.

The source and header files added to LAMMPS are
``dump_atom_smartsim.cpp`` and ``dump_atom_smartsim.h``,
respectively.  The files should be placed in a directory called
``USER-SMARTSIM`` inside of the ``lammps/src`` directory to
conform to LAMMPS developer standards.

The majority of code in ``dump_atom_smartsim.cpp``
is devoted to placing atom information from
LAMMPS data structures into arrays that can be sent to
the SmartSim database.  Aside from that data manipulation,
there are only four unique client API calls needed
to send all of the tensor and metadata to the SmartSim
experiment database:

1) ``Client`` constructor to create the ``Client`` object
2) ``DataSet`` constructor to create the ``DataSet`` object
2) ``DataSet.add_tensor()`` to add the atom ID and atom type
   tensors to the DataSet
3) ``DataSet.add_meta_scalar()`` to add simulation metadata
   to the DataSet
4) ``Client.put_dataset()`` to send the ``DataSet`` to the
   SmartSim experiment database.

For completeness, the code snippet below has been extracted from
the LAMMPS source files to show the exact
client API calls that are necessary to send the atom id,
atom type, atom coordinates, and simulation metadata
to the SmartSim database.

.. code-block:: cpp

  /* Construct SILC Client object
  */
  SILC::Client client(true);

  /* Construct DataSet object with unique
  name based on user prefix, MPI rank, and
  timestep
  */
  SILC::DataSet dataset(this->_make_dataset_key());

  /* Add a "domain" metadata field to the DataSet to hold information
  about the simulation domain.
  */
  dataset.add_meta_scalar("domain", &(domain->boxlo[0]), SILC::MetaDataType::dbl);
  dataset.add_meta_scalar("domain", &(domain->boxhi[0]), SILC::MetaDataType::dbl);
  dataset.add_meta_scalar("domain", &(domain->boxlo[1]), SILC::MetaDataType::dbl);
  dataset.add_meta_scalar("domain", &(domain->boxhi[1]), SILC::MetaDataType::dbl);
  dataset.add_meta_scalar("domain", &(domain->boxlo[2]), SILC::MetaDataType::dbl);
  dataset.add_meta_scalar("domain", &(domain->boxhi[2]), SILC::MetaDataType::dbl);

  /* Add a "triclinic" metadata field to the DataSet to indicate
  if the triclinic boolean is true in the simulation.
  */
  dataset.add_meta_scalar("triclinic", &(domain->triclinic), SILC::MetaDataType::int64);

  /* If the triclinic boolean is true, add triclinic metadata
  fields to the DataSet.
  */
  if(domain->triclinic) {
    dataset.add_meta_scalar("triclinic_xy", &(domain->xy), SILC::MetaDataType::dbl);
    dataset.add_meta_scalar("triclinic_xz", &(domain->xz), SILC::MetaDataType::dbl);
    dataset.add_meta_scalar("triclinic_yz", &(domain->yz), SILC::MetaDataType::dbl);
  }

  /* Add a "scale_flag" metadata field ot the DataSet to indicate
  if the simulation scale_flag is true or false.
  */
  dataset.add_meta_scalar("scale_flag", &scale_flag, SILC::MetaDataType::int64);

  /* Perform internal LAMMPS output preprocessing.
  This code is omitted for brevity.
  */

  std::vector<size_t> tensor_length;
  tensor_length.push_back(n_local);

  //Add atom ID tensor to the DataSet
  this->_pack_buf_into_array<int>(data_int, buf_len, 0, n_cols);
  dataset.add_tensor("atom_id", data_int, tensor_length,
                      SILC::TensorType::int64, SILC::MemoryLayout::contiguous);

  //Add atom type tensor to the DataSet
  this->_pack_buf_into_array<int>(data_int, buf_len, 1, n_cols);
  dataset.add_tensor("atom_type", data_int, tensor_length,
                      SILC::TensorType::int64, SILC::MemoryLayout::contiguous);

  //Add atom x position  tensor to the DataSet
  this->_pack_buf_into_array<double>(data_dbl, buf_len, 2, n_cols);
  dataset.add_tensor("atom_x", data_dbl, tensor_length,
                      SILC::TensorType::dbl, SILC::MemoryLayout::contiguous);

  //Add atom y position  tensor to the DataSet
  this->_pack_buf_into_array<double>(data_dbl, buf_len, 3, n_cols);
  dataset.add_tensor("atom_y", data_dbl, tensor_length,
                      SILC::TensorType::dbl, SILC::MemoryLayout::contiguous);

  //Add atom z position tensor to the DataSet
  this->_pack_buf_into_array<double>(data_dbl, buf_len, 4, n_cols);
  dataset.add_tensor("atom_z", data_dbl, tensor_length,
                      SILC::TensorType::dbl, SILC::MemoryLayout::contiguous);

  /*Add "image_flag" metadata field to the DataSet to indicate
  if the image_flag boolean is true of false in the simulation.
  */
  dataset.add_meta_scalar("image_flag", &image_flag, SILC::MetaDataType::int64);
  if (image_flag == 1) {
    //Add atom ix image tensor to the DataSet
    this->_pack_buf_into_array<int>(data_int, buf_len, 5, n_cols);
    dataset.add_tensor("atom_ix", data_int, tensor_length,
                        SILC::TensorType::int64, SILC::MemoryLayout::contiguous);

    //Add atom iy image tensor to the DataSet
    this->_pack_buf_into_array<int>(data_int, buf_len, 6, n_cols);
    dataset.add_tensor("atom_iy", data_int, tensor_length,
                        SILC::TensorType::int64, SILC::MemoryLayout::contiguous);

    //Add atom iz image tensor to the DataSet
    this->_pack_buf_into_array<int>(data_int, buf_len, 7, n_cols);
    dataset.add_tensor("atom_iz", data_int, tensor_length,
                        SILC::TensorType::int64, SILC::MemoryLayout::contiguous);
  }

  /* Send the DataSet to the SmartSim experiment database
  */
  client.put_dataset(dataset);

The ``atom/smartsim`` dump style generates a key for the
``DataSet`` based on the simulation time step number, the
MPI rank of the process, and the key prefix provided
by the user in the LAMMPS input file.  This is shown in the
above code snippet with the call to ``_make_key()``
function that has been implemented in LAMMPS.  The ``_make_key()``
function is shown below as an example of key generation, but
the other applications  will require the user to
write their own key generation scheme. Note that in the current
implementation of the ``atom/smartsim`` dump style, each quantity is
sent to the SmartSim database as a 1D tensor in the same order
as the 1D tensor enumerating atom IDs and each MPI process
sends its own data to the database.  However, ``Client`` and
``DataSet`` array functions support n-dimensional tensors
if the user wanted to combine all of the quantities into a
single n-dimensional tensor.

.. code-block:: cpp

  std::string DumpAtomSmartSim::_make_dataset_key()
  {
    // create database key using the var_name

    int rank;
    MPI_Comm_rank(world, &rank);

    std::string prefix(filename);
    std::string key = prefix + "_rank_" + std::to_string(rank) +
      "_tstep_" + std::to_string(update->ntimestep);
    return key;
}

Compiling with the SmartSim Client
----------------------------------

To use the SILC client at application runtime,
the client will need to be linked with your
application and the header file directories
included with application include search paths.
To build the SILC client with your
application, you will need to include the following items:

1)  ``smartsim/utils/protobuf/`` and
    ``smartsim/clients`` should be added to your
    include directories when compiling.  It is best
    practice to use the environment variable ``SILC_INSTALL_PATH``
    that is set when sourcing ``setup_env.sh`` to
    point to the top level SILC directory when
    adding these directories.  In CMAKE, this could be
    implemented as shown in the code snippet below.

.. code-block:: cmake

  include_directories($ENV{SILC_INSTALL_PATH}/clients/)
  include_directories($ENV{SILC_INSTALL_PATH}/utils/protobuf/)

2)  ``hiredis``, ``redis-plus-plus``, and ``protobuf``
    include directories should be added to your make file include paths.
    These packages and include directories are installed
    during the SILC installation and can be referenced
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

3)  To avoid including the source files as part of the application
    build, the SILC static library should be linked with the
    application.  To build the SILC static library, the
    command ``make lib`` should be first executed before
    sourcing ``setup_env.sh`` in the top level SILC directory.
    After building the static library, the library location can be
    referenced using the ``SILC_INSTALL_PATH`` environment variable.
    For those applications that use CMAKE, the code snippet
    below shows how a user can include the static library.

.. code-block:: cmake

   string(CONCAT SILC_LIB_PATH $ENV{SILC_INSTALL_PATH} "/build")
   find_library(SILC_LIB silc PATHS ${SILC_LIB_PATH} REQUIRED)

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
   find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH REQUIRED)

   # Use environment variable PROTOBUF_INSTALL_PATH to set
   # location of protobuf lib
   string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
   find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH REQUIRED)

   # Use environment variable REDISPP_INSTALL_PATH to set
   # location of redis-plus-plus lib
   string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
   find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} NO_DEFAULT_PATH REQUIRED)

   # Store the three libraries in a variable CLIENT_LIBRARIES for easy linking
   set(CLIENT_LIBRARIES ${SILC_LIB} ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

For clarity, the aforementioned code snippets have been folded into a
working CMAKE file shown below that would build the SILC client
into an application called ``my_application``.

.. code-block:: cmake

  project(Example)

  set(CMAKE_BUILD_TYPE Release)

  cmake_minimum_required(VERSION 3.10)

  SET(CMAKE_CXX_STANDARD 17)

  # Add the SILC Client include directories using the
  # SILC_INSTALL_PATH environment variable
  include_directories($ENV{SILC_INSTALL_PATH}/include)
  include_directories($ENV{SILC_INSTALL_PATH}/utils/protobuf)

  # Add the SILC static libarary using the SILC_INSTALL_PATH
  # enviroment variable
  string(CONCAT SILC_LIB_PATH $ENV{SILC_INSTALL_PATH} "/build")
  find_library(SILC_LIB silc PATHS ${SILC_LIB_PATH} REQUIRED)

  # Add the third-party package include paths to the
  # project using the environment variables provided by SILC
  string(CONCAT HIREDIS_INCLUDE_PATH $ENV{HIREDIS_INSTALL_PATH} "/include/")
  string(CONCAT PROTOBUF_INCLUDE_PATH $ENV{PROTOBUF_INSTALL_PATH} "/include/")
  string(CONCAT REDISPP_INCLUDE_PATH $ENV{REDISPP_INSTALL_PATH} "/include/")
  include_directories(${HIREDIS_INCLUDE_PATH})
  include_directories(${PROTOBUF_INCLUDE_PATH})
  include_directories(${REDISPP_INCLUDE_PATH})

  # Use environment variable HIREDIS_INSTALL_PATH to set
  # location of hiredis lib
  string(CONCAT HIREDIS_LIB_PATH $ENV{HIREDIS_INSTALL_PATH} "/lib")
  find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH REQUIRED)

  # Use environment variable PROTOBUF_INSTALL_PATH to set
  # location of protobuf lib
  string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
  find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH REQUIRED)

  # Use environment variable REDISPP_INSTALL_PATH to set
  # location of redis-plus-plus lib
  string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
  find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} NO_DEFAULT_PATH REQUIRED)

  # Store the three third-party libraries in a variable
  # CLIENT_LIBRARIES for easy linking
  set(CLIENT_LIBRARIES ${SILC_LIB} ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

  # Build my application
  add_executable(my_application
 	my_application.cpp
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
2) Include the ``USER-SMARTSIM`` package CMAKE file
   (line 340)`.  This CMAKE file will include the SILC
   libraries and header files if the USER-SMARTSIM
   package is requested by the user.

It is worth reiterating that the CMAKE examples presented here
rely on environment variables set by
the SILC ``setup_env.sh`` script.  Therefore,
``setup_env.sh`` should be run before trying to compile
with the SILC client.

To build LAMMPS with the aforementioned
``USER-SMARTSIM`` package and MPI capabitlies,
the following cmake command can be used:

.. code-block:: bash

   cmake ../ -DBUILD_MPI=yes -DPKG_USER-SMARTSIM=ON

The LAMMPS binary location should be added
to the PATH environment variable so that the
SmartSim experiment can find it.  Additionally,
it is recommended that the ``stable`` branch
of LAMMPS be used for this tutorial.


Experiment Setup
================

In this tutorial, the SmartSim experiment consists of a
SmartSim model entity for the LAMMPS simulation and a
SmartSim mdoel entity to intake and plot atom position
information from the SmartSim database.  The experiment
is configured to utilize the Slurm launcher
and a 3 node KeyDB redis cluster database.  The
SmartSim experiment script is shown below with
comments to explain the experiment progression.
It is worth noting that the inclusion of the
SILC client in LAMMPS does not alter the
typical experiment flow that has been described in
other tutorials.  In fact, no details of the
C++ client utiliation in LAMMPS are necessary
in the SmartSim experiment script.


.. code-block:: python

from smartsim import Experiment, slurm
from os import environ
import os

# Define resource variables for models,
# scripts, and orchestrator
lammps_compute_nodes = 2
lammps_ppn = 2
db_compute_nodes = 3
analysis_compute_nodes = 1

total_nodes = lammps_compute_nodes + \
              db_compute_nodes + \
              analysis_compute_nodes

# Retrieve Slurm allocation for the experiment
alloc = slurm.get_slurm_allocation(nodes=total_nodes)

# Create a SmartSim Experiment using the default
# Slurm launcher backend
experiment = Experiment("lammps_experiment")

# Define the run settings for the LAMMPS model that will
# be subsequently created.
lammps_settings = {
    "nodes": lammps_compute_nodes,
    "ntasks-per-node" : lammps_ppn,
    "executable": "lmp",
    "exe_args": "-i in.melt",
    "alloc": alloc}

# Define the run settings for the Python analysis script
# that will be subsequently created
analysis_settings = {
    "nodes": analysis_compute_nodes,
    "executable": "python",
    "exe_args": f"data_analysis.py --ranks={lammps_compute_nodes*lammps_ppn} --time=250",
    "alloc": alloc}

# Create the LAMMPS SmartSim model entity with the previously
# defined run settings
m1 = experiment.create_model("lammps_model", run_settings=lammps_settings)

# Attach the simulation input file in.melt to the entity so that
# the input file is copied into the experiment directory when it is created
m1.attach_generator_files(to_copy=["./in.melt"])

# Create the analysis SmartSim entity with the
# previously defined run settings
m2 = experiment.create_model("lammps_data_processor",run_settings=analysis_settings)

# Attach the analysis script to the SmartSim node entity so that
# the script is copied into the experiment directory when the
# experiment is generated.
m2.attach_generator_files(to_copy=["./data_analysis.py"])

# Create the SmartSim orchestrator object and database using the default
# database cluster setting of three database nodes
orc = experiment.create_orchestrator(db_nodes=db_compute_nodes,
                                     overwrite=True, alloc=alloc)

# Generate the experiment directory structure and copy the files
# attached to SmartSim entities into that folder structure.
experiment.generate(m1, m2, orc, overwrite=True)

# Start the model and orchestrator
experiment.start(m1, orc, summary=True)

# Start the data analysis script after the model is complete
experiment.start(m2, summary=True)

# When the model and node are complete, stop the
# orchestrator with the stop() call which will
# stop all running jobs when no entities are specified
experiment.stop(orc)

# Release our system compute allocation
# experiment.release()
slurm.release_slurm_allocation(alloc)

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
script, SILC Python client API functions are used to retrieve
data from the database.  Specifically, ```get_dataset`` and
``get_tensor`` are used to retreive the data. It is worth
noting that the function names in the Python client are nearly
identical to the C, Fortran, and C++ client function names,
but the function parameters may differ.

The important steps in the analysis script below
that are applicable to all Python scripts that send and receive
data from the SmartSim database are as follows:

1)  Import the SILC ``Client`` and ``DataSet``
    object for use in the script with
    ``from silc import Client, Dataset``
2)  Initialize a SILC ``Client`` object with
    ``client = Client()`` or
    ``client = Client(cluster=True)``.  The optional argument
    ``cluster`` is by default ``False``, and indicates whether
    or not a cluster of database nodes is being used.  The
    SmartSim experiment database address is typically
    determined through the environment variable ``SSDB``,
    but it can be specifically manually in the ``Client``
    constructor.
3)  Retrieve each LAMMPS ``DataSet`` that was saved
    in the SmartSim experiment database with
    ``Client.get_dataset()``.
3)  Use SILC ``DataSet`` member function ``get_tensor``
    to retrieve each ``DataSet`` tensor corresponding
    to atom ID, atom type, and atom position.
In this particular example, the atom positions that are
fetched from the database are plotted on a three-dimensional
plot.  As expected, the molecules should be uniformly
distributed in the domain.

.. code-block:: python

  import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from silc import Client, Dataset

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

        dataset_key = f"atoms_rank_{i}_tstep_{t_step}"

        print(f"Retrieving DataSet {dataset_key}")

        dataset = client.get_dataset(dataset_key)

        atom_id.extend(dataset.get_tensor("atom_id"))
        atom_type.extend(dataset.get_tensor("atom_type"))
        atom_x.extend(dataset.get_tensor("atom_x"))
        atom_y.extend(dataset.get_tensor("atom_y"))
        atom_z.extend(dataset.get_tensor("atom_z"))

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
