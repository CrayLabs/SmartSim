
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
written which will send 3D layer thicknesses averaged over of every modelled
day. This module contains two subroutines 1) initializes and stores the SmartSim client on each processor 2) Averages the 3D arrays and determines when to send data to the database

``initialize_smartsim_connector`` is called during the initialization routine of MOM6.
from multiple processors. First, the rank of the PE will be stored and used
when constructing a key to be sent to the database. Without this, every PE
would send their own arrays using the same key resulting in data being
overwritten. Second, a two integer array is sent to the database containing
the index within the global (non-decomposed) domain corresponding to the
first element of the arrays containing temperature and salinity.

.. code-block:: fortran

    !> Initialize the SmartSim client and send any static metadata
    subroutine initialize_smartsim_connector(CS, G)
      type(smartsim_type), pointer,  intent(inout) :: CS !< Control structure for the smartsim connector
      type(ocean_grid_type),         intent(in)    :: G  !< Contains grid metrics

      integer, dimension(2) :: global_start_indices
      global_start_indices(1) =  G%isg
      global_start_indices(2) =  G%jsg

      ! Allocate the control structure that MOM6 uses to configure and store
      ! data in every module
      allocate(CS)
      ! Initialize and store the SmartSim Client
      CS%smartsim_client = init_ssc_client()
      ! Store the zero-padded processor element as a unique ID
      write(CS%unique_id,'(I5.5)') pe_here()
      ! Send the indices where the first element in each array exists in the
      ! global index
      call put_array(CS%smartsim_client, CS%unique_id//"_rank-meta", global_start_indices)
      ! Initialize the accumulated time between sends to 0
      CS%accumulated_time = 0.

      ! Allocate the arrays used to store the averages of layer thickness
      ALLOC_(CS%h_avg(G%isd:G%ied,G%jsd:G%jed,G%ke)); CS%h_avg(:,:,:) = 0.

    end subroutine initialize_smartsim_connector

The second subroutine ``send_variables_to_database`` is called at the end of the
thermodynamics calculations in the model. This routine handles the averaging
of the arrays and when the specified averaging interval is reached, sends the
thicknesses to the arrays. Different time levels are stored in the database
by assigning a prefix to each key based on the MPI rank saved during the
initialization and the current day of the model.

.. code-block:: fortran

    !> Send the current state of temperature and salinity to the database
    subroutine send_variables_to_database(CS, time, dt, h)
      type(smartsim_type),      intent(inout) :: CS   !< Control structure for the smartsim connector
      type(time_type),          intent(in   ) :: time !< Current time of the model
      real,                     intent(in   ) :: dt   !< Length of the timestep
      real, dimension(:,:,:),   intent(in   ) :: h    !< 3D array of layer thicknesses

      character(len=10) :: datestring
      character(len=16) :: prefix
      real :: wt
      integer :: month, day, year, minute, hour, second

      ! Update the accumulated time
      CS%accumulated_time = CS%accumulated_time + dt

      ! Update the arrays containing averages
      wt = dt/CS%averaging_interval
      if (CS%send_h) CS%h_avg(:,:,:) = CS%h_avg(:,:,:) + h(:,:,:)*wt

      ! Check whether it's time to send data to the database
      if (CS%accumulated_time == CS%averaging_interval) then

        ! Construct a key prefix based on the time and the pe rank
        ! Write the time as as string (YYYY-MM-DD)
        call get_date(Time, year, month, day, hour, minute, second)
        write(datestring,'(I4.4,A,I2.2,A,I2.2)') year, '-', month, '-', day
        write(prefix,'(A,A,A,A)') CS%unique_id, '_', datestring, '_'

        ! Send any of the requested fields
        if (CS%send_h) then
          call put_array(CS%smartsim_client, prefix//'h', CS%h_avg)
          CS%h_avg(:,:,:) = 0.
        endif
      endif
    end subroutine send_variables_to_database

Compiling the Fortran Client with MOM6
--------------------------------------
An instrumented version of MOM6 can be found at `599773
<https://github.com/ashao/MOM6/commit/599773fa53058e30e4167216c8cf7f79a75b255f>`_
and is compatible with the full MOM6-examples (`0de2f77
<https://github.com/NOAA-GFDL/MOM6-examples/commit/0de2f77e8b4a78fcd5f5b7f9ea90f0ccf37f49d9>`_).
Compiling MOM6 with SmartSim requires slight modifications to the standard
compilation workflow described on the `MOM6 wiki
<https://github.com/NOAA-GFDL/MOM6-examples/wiki>`_.

First clone the MOM6-examples directory (this needs to be done recursively to
get all the submodules). The root of this directory will be referred to as ``$MOM6EXAMPLES_PATH$``

.. code-block:: bash

    git clone --recursive https://github.com/NOAA-GFDL/MOM6-examples.git
    cd MOM6-examples
    git checkout 0de2f77e8b4a78fcd5f5b7f9ea90f0ccf37f49d9

From here add a new remote for MOM6 and checkout the correct commit

.. code-block:: bash

    cd src/MOM6
    git remote add smartsim https://github.com/ashao/MOM6.git
    git fetch smartsim
    git checkout smartsim/smartsim
    cd ../../

The current directory should now again be at the MOM6-examples root. Copy the build directory from $SMARTSIMHOME from this MOM6 example directory

.. code-block:: bash

    cp -r $SMARTSIMHOME/examples/MOM6/build ./

This directory contains a modified version of ``list_paths`` (used to find
all source files matching given extensions) to support C++ source code files
(``.cc``). Additionally, a makefile template ``ncrc-gnu.mk`` compatible with
the ``PrgEnv-gnu`` environments on XC machines has been added. The
modifications to ``INCLUDE`` and ``LDFLAGS`` ensure that SmartSim
dependencies are properly linked in.

The following command is then used to compile what the ``FMS`` infrastructure library used by MOM6.

.. code-block:: bash

    mkdir -p build/gnu/shared/repro/
    (cd build/gnu/shared/repro/; rm -f path_names; \
      ../../../../build/list_paths -l ../../../../src/FMS; \
      ../../../../src/mkmf/bin/mkmf -t ../../../../build/gnu/ncrc-gnu.mk -p libfms.a -c "-Duse_libMPI -Duse_netCDF -DSPMD" path_names)
    (cd build/gnu/shared/repro/; source ../../env; make NETCDF=3 REPRO=1 libfms.a -j)

After that has finished building, MOM6 itself can be compiled (complete with
the SmartSim clients).

.. code-block:: bash

    mkdir -p build/gnu/ocean_only/repro/
    (cd build/gnu/ocean_only/repro/; rm -f path_names; \
      ../../../../build/list_paths -l ./ ../../../../src/MOM6/{config_src/dynamic,config_src/external,config_src/solo_driver,src/{*,*/*}} $SMARTSIMHOME/smartsim/{clients,utils/protobuf} ; \
      ../../../../src/mkmf/bin/mkmf -t ../../../../build/gnu/ncrc-gnu.mk -o "-I../../shared/repro/" -p MOM6 -l "-L../../shared/repro -lfms" -c '-Duse_libMPI -Duse_netCDF -DSPMD' path_names)
    (cd build/gnu/ocean_only/repro/; source ../../env; make NETCDF=3 REPRO=1 MOM6 -j)

The primary difference between the instructions on the MOM6 wiki is the
inclusion of the SmartSim clients directory. A successful compile results in a MOM6 executable at

.. code-block:: bash

    /path/to/MOM6-examples/build/gnu/ocean_only/repro/MOM6

Start Jupyter environment
==============================

First, start a Jupyter notebook or Jupyter lab server on a compute node and
connect to it. This can easily be done with Cray Urika-XC for analysis
scale that needs to be done at scale. Alternatively, JupyterHub can be used or
a slurm job can be submitted manually port forwards from the compute node.

On some machines, compute nodes cannot interact with the cluster scheduler.
In this case, a command server will need to be started within the notebook to
handle the parts of the SmartSim experiment that need to interact with the
scheduler or resource allocations.

Running and Analyzing MOM6 from Jupyter
=======================================

The included Jupyter notebook ``interactive_MOM6.ipynb`` has three main parts
1) setting up a MOM6 experiment in SmartSim 2) interactively plotting results
as they become available and 3) performing trend analysis on model output.

Setting up the experiment
-------------------------

MOM6 is used setup to run an eddy-resolving version of the two-layer
double-gyre model that is a useful analogue to real ocean gyres. This example
requires no data inputs and can be run on a single node. See Section 1 of the
notebook for details and actual code

Communicating with the simulation
---------------------------------

In addition to setting up the experiment, the notebook is also used to
interactively visualize output from the model. To do this the
appropriate environment variables must be set namely, the address of the
database. The following both accomplishes both

.. code-block:: python

  os.environ["SSDB"] = experiment.get_db_address()[0]
  client = Client(cluster=True)

Now the notebook has access to all of Python client's methods including
receiving arrays of different types.

Visualization and analysis
--------------------------

MOM6 follows a domain decomposition strategy for parallelization. Each
processor works on a logically-rectangular subdomain of the model with MPI
calls to synchronize halos (also termed ghost cells) when necessary. Most
visualization and analysis requires the entire global domain. For models
following a similar scheme to MOM6, all that's needed is the location of
the subdomain in the global array. This information was sent to the database
on initialization from every MPI rank and stored in the database as
``{rank}_array-meta``.

The reconstruction script loops over all the ranks and retrieves arrays from
each subdomain. It then allocates and fills in the global based on both the
metadata and MOM6's daily-averaged layer thicknesses.

The ``get_array`` commands can be specified to 'wait' until the key is
available in the database. For example in the following example, the
following code will retrieve an array with the given key over all 32 ranks
and from the second month of the model integration corresponding to the
layer thicknesses (``h``).

.. code-block:: python

  field_name = 'h'
  layer_thickness = {}
  for rank in range(32):
    rank_id = f'{rank:05d}'
    layer_thickness[rank_id] = {}
    for day in range(30):
      timestring = f'0001-02-{day:02d}'
      layer_thickness[rank_id][timestring] = client.get_array_nd_float64(
                        f'{rank_id}_{timestring}_{field_name}', wait=True)

This call will block further execution until completion.

The arrays retrieved from the database are numpy arrays and are thus
compatible with a variety of packages.

The notebook includes an example performs simple timeseries analysis by
first reconstructing 30 days of model output and then using ``numpy.polyfit``
to calculate the trends in layer thickness.

