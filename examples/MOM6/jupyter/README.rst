
***********************************************
Interacting with Fortran Simulations in Jupyter
***********************************************


Users can interact with a SmartSim experiment via Jupyter notebooks for online
analysis and visualization. Simulations written in Fortran, C, C++ or Python
can use the SmartSim clients to stream data out of the model such that
the simulation data can be consumed by an analysis pipeline, or in this
case, by a Jupyter notebook

This example will show how to connect a MOM6 simulation, written in Fortran,
to a Jupyter notebook so that fields from the model can be plotted in real
time as the model is running.

Modular Ocean Model 6 (MOM6)
============================

MOM6 is a 3-dimensional general circulation model- a class of computational
fluid dynamics models specifically designed to solve geophysical flows. A
modified version of the Navier-Stokes equations are solved using a
finite-volume discretization and logically rectangular cells on a structured
grid or unstructured mesh. When run in parallel, the domain is decomposed and
distributed among the available processing elements. The halos (i.e. ghost
cells) are updated via MPI.

Using the SmartSim Fortran Client with MOM6
===========================================

In this section we will describe how to compile the SmartSim Fortran client
into the MOM6 codebase and how we embedded the client into the MOM6 codebase.

How the Fortran Client was Embedded into MOM6
---------------------------------------------

Every module within MOM6 stores its own configuration and persistent data in
a "control structure", a derived type. In this example, a new module is
written which will send 3D temperature and salinity at the end
of every modelled day.

.. code-block:: fortran

    module smartsim_connector
        use client_fortran_api, only :: put_array, init_ssc_client
        use iso_c_binding,      only :: c_ptr
        implicit none; private

        type smartsim_connector_type, public :: private
            type(c_ptr)        :: smartsim_client    !< Pointer to an initialized Smartsim Client
            character(len=255) :: unique_id          !< A string used to identify this PE
            logical            :: send_temp = .true. !< If true, send temperatrue
            logical            :: send_salt = .true. !< If true, send salinity
        end type smartsim_connector_type

        public :: initialize_smartsim_connector
        public :: send_variables_to_database

        contains

        !> Initalize the SmartSim client and send any static metadata
        subroutine initialize_smartsim_connector(CS, pe_rank, isg, jsg)
            type(smartsim_connector_type), intent(in) :: CS      !< Control structure for the smartsim connector
            integer,                       intent(in) :: pe_rank !< The rank of the processor, used as an identifier
            integer,                       intent(in) :: isg     !< Index within the global domain along the i-axis
            integer,                       intent(in) :: jsg     !< Index within the global domain along the i-axis

            allocate(CS)
            CS%smartsim_client = init_ssc_client()
            write(CS%unique_id,'(I5.5)') pe_rank
            call put_array(CS%smartsim_client, CS%unique_id//"_rank-meta", REAL([/ isg,jsg /]))

        end subroutine initialize_smartsim_connector

        !> Send the current state of temperature and salinity to the database
        subroutine send_variables_to_database(CS, temp, salt)
            type(smartsim_connector_type), intent(in) :: CS   !< Control structure for the smartsim connector
            real, dimension(:,:,:),        intent(in) :: temp !< 3D array of ocean temperatures
            real, dimension(:,:,:),        intent(in) :: salt !< 3D array of ocean salinity

            if (CS%send_temp) call put_array(CS%smartsim_client, CS%unique_id//"_temp", temp)
            if (CS%send_salt) call put_array(CS%smartsim_client, CS%unique_id//"_salt", salt)
        end subroutine send_variables_to_database

    end module smartsim_connector

During model intialization, ``initialize_smartsim_connector`` is called. The
SmartSim client is initialized in the same way as in the previous toy
example, however an additional two steps are taken because data will be sent
from multiple processors. First, the rank of the PE will be stored and used
when constructing a key to be sent to the database. Without this, every PE
would send their own arrays using the same key resulting in data being
overwritten. Second, a two integer array is sent to the database containing
the index within the global (non-decomposed) domain corresponding to the
first element of the arrays containing temperature and salinity.

In ``send_variables_to_database``, the arrays containing temperature and
salinity are sent to the database. The unique id is again added to the key
to ensure that data is not overwritten by another processor.



Compiling the Fortran Client with MOM6
--------------------------------------


Put information in about compiling with SmartSim Fortran client that
includes
 - commit link to the commit used
 - step by step compilation instructions
 - gotchas



Spin up a Jupyter Environment
==============================

First, start a Jupyter notebook or Jupyter lab server on a compute node and
connect to it. This can easily be done with Cray Urika-XC for analysis
scale that needs to be done at scale. Alternatively, JupyterHub can be used or
a slurm job can be submitted manually port forwards from the compute node.

In most of those cases, a command server will need to be started within the
notebook to handle the parts of the SmartSim experiment that need to interact
with the scheduler or resource allocations.


Running and Analyzing MOM6 from Jupyter
=======================================

Setting up the Experiment
-------------------------

describe the experiemnt and fields
show python code

Communicating with the Simulation
---------------------------------

How use the python client to get fields
show python code

Analysis
--------

how to perform some analysis including the domain reconstruction
and ploting and hopefully some real analysis.