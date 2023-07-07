*****************************
Integrating into a Simulation
*****************************

========
Overview
========

This document provides some general guidelines to integrate the SmartRedis client
into existing simulation codebases. Developers of these simulation codebases will
need to identify the exact places to add the code; generally SmartRedis calls
will only need to be added in two places:

1. Initialization
2. Main loop

==============
Initialization
==============

+++++++++++++++++++
Creating the client
+++++++++++++++++++

The SmartRedis client must be initialized before it can be used to communicate
with the orchestrator. In the C++ and Python versions of the clients, this is done
when creating a new client. In the C and Fortran client an `initialize`
method must be called.

C++::

    #include "client.h"
    SmartRedis::Client client(use_cluster);

Python::

    from smartredis import Client
    client = Client(use_cluster)

Fortran::

    use smartredis_client, only : client_type
    include "enum_fortran.inc"

    type(client_type) :: client
    return_code = client%initialize(use_cluster)
    if (return_code .ne. SRNoError) stop 'Error in initialization'

C::

    #include "client.h"
    #include "sr_enums.h"
    void* client = NULL;
    return_code = SmartRedisCClient(use_cluster, &client)
    if (return_code != SRNoError) {
        return -1
    }

All these methods only have one configurable parameter -- indicated in the
above cases by the variable `use_cluster`. If this parameter is true,
then the client expects to be able to communicate with an orchestrator with
three or more shards.

++++++++++++++++++++++++++++++++++++++++++
(Parallel Programs): Creating unique names
++++++++++++++++++++++++++++++++++++++++++

For parallel applications, each rank or thread that is communicating with
the orchestrator will likely need to create a unique prefix for names to prevent
another rank or thread inadvertently overwriting data. This prefix should be used
for when creating the name of a tensor, dataset, and model that needs to be unique
to a given rank. (Note: for models run within SmartSim, Additional prefixing may
done by the client when running an ensemble and/or multiple data sources).
Any identifier can be used, though typically the MPI rank number (or equivalent
identifier) is a useful, unique number.

C++::

    const std::string name_prefix << std::format("{:06}_", *rank_id);

Python::

    name_prefix = f"{rank_id:06d}_"

Fortran::

    character(len=12) :: name_prefix
    write(name_prefix,'(A,I6.6)') rank_id

C::

    char[7] name_prefix;
    name_prefix = sprintf(name_prefix", "%06d\0", *rank_id);

++++++++++++++++++++++++++
Storing scripts and models
++++++++++++++++++++++++++

The last task that typically needs to be done is to store models or scripts
that will be used later in the simulation. When using a clustered orchestrator,
this only needs to be done by one client (unless each rank requires a different
model or script). MPI rank 0 is often a convenient choice to set models and
scripts.

C++::

    if (root_client) {
        client.set_model_from_file(model_name, model_file, backend, device)
    }

Python::

    if root_client:
        client.set_model_from_file(model_name, model_file, backend, device)

Fortran::

    if (root_client) return_code = client%set_model_from_file(model_name, model_file, backend, device)
    if (return_code .ne. SRNoError) stop 'Error setting model'

C::

    if (root_client) {
        return_code = client.set_model_from_file(client, model_name, model_file, backend, device)
        if (return_code != SRNoError) {
            return -1
        }
    }

=========
Main loop
=========

Within the main loop of the code (e.g. every timestep or iteration of a solver),
the developer typically uses the SmartRedis client methods to implement a workflow which
may include receiving data, sending data, running a script or model, and/or
retrieving a result. These workflows are covered extensively in the walkthroughs
for the Fortran, C++, and python clients and the integrations with MOM6, OpenFOAM,
LAMMPS, and others.

Generally though, developers are advised to:

1. Find locations where file I/O would normally happen and either augment
   or replace code to use the SmartRedis client and store the data in the
   orchestrator
2. Use the `name_prefix` created during initialization to avoid accidental
   writes/reads from different clients
3. Use the SmartSim `dataset` type when using clients representing decomposed
   subdomains to make the retrieval/use of the data more performant

============
Full example
============

The following pseudocode is used to demonstrate various aspects of instrumenting an
existing simulation code with SmartRedis. This code is representative of solving
the time-evolving heat equation. but we will augment it using an ML model to
provide a preconditioning step each iteration and post the state of the simulation
to the orchestrator. ::

    program main

        ! Initialize the model, setup MPI, communications, read input files
        call initialize_model(temperature, number_of_timesteps)

        main_loop: do i=1,number_of_timesteps

            ! Write the current state of the simulation to a file
            call write_current_state(temperature)

            ! Call a time integrator to step the temperature field forward
            call timestep_simulation(temperature)

        enddo
    end program main

Following the guidelines from above, the first step is to initialize the client
and create a unique identifier for the given processor. This should be done
within roughly the same portion of the code where the rest of the model
performs the initialization of other components. ::

    ! Import SmartRedis modules
    use, only smartredis_client : client_type
    ! Include all fortran enumerators especially for error checking
    include "enum_fortran.inc"

    ! Declare a new variable called client and a string to create a unique
    ! name for names
    type(client_type) :: smartredis_client
    character(len=7)  :: name_prefix
    integer :: mpi_rank, mpi_code, return_code

    ! Note adding use_cluster as an additional runtime argument for SmartRedis
    call initialize_model(temperature, number_of_timesteps, use_cluster)
    return_code = smartredis_client%initialize(use_cluster)
    if (return_code .ne. SRNoError) stop 'Error in init'
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, mpi_code)
    ! Build the prefix for all tensors set in this model
    write(name_prefix,'(I6.6,A)') mpi_rank, '_'

    ! Assume all ranks will use the same machine learning model, so no need to
    ! add the prefix to the model name
    if (mpi_rank==0) then
        return_code = set_model_from_file("example_model_name", "path/to/model.pt", "TORCH", "gpu")
        if (return_code .ne. SRNoError) stop 'Error in setting model'
    endif


Next, add the calls in the main loop to send the temperature to the orchestrator ::

    character(len=30), dimension(1) :: model_input, model_output

    main_loop: do i=1,number_of_timesteps

        ! Write the current state of the simulation to a file
        call write_current_state(temperature)
        model_input(1) = name_prefix//"temperature"
        model_output(1) = name_prefix//"temperature_out"
        return_code = smartredis_client%put_tensor(model_input(1), temperature)
        if (return_code .ne. SRNoError) stop 'Error in putting tensor'

        ! Run the machine learning model
        return_code = smartredis_client%run_model("example_model_name", model_input, model_output)
        ! The following line overwrites the prognostic temperature array
        return_code = smartredis_client%unpack_tensor(model_output(1), temperature)
        if (return_code .ne. SRNoError) stop 'Error in retrieving tensor'

        ! Call a time integrator to step the temperature field forward
        call timestep_simulation(temperature)

    enddo

This model will now use the client every timestep to put a
temperature array in the orchestrator, instruct the orchestrator to call
a machine learning model for prediction/inference, and unpack the resulting
inference into the existing temperature array. For more complex examples,
please see some of the integrations in the SmartSim Zoo or feel free to
contact the team at CrayLabs@hpe.com